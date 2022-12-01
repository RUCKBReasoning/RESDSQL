import os
import json
import math
import torch
import transformers
import argparse
import torch.optim as optim

from tqdm import tqdm
from tokenizers import AddedToken
from utils.ranker_metric.evaluator import cls_metric, table_ranking_metric, column_ranking_metric
from torch.utils.data import DataLoader
from transformers import RobertaTokenizerFast
from utils.models import MyRanker
from utils.ranker_loss import RankerLoss
from transformers.trainer_utils import set_seed
from torch.utils.tensorboard import SummaryWriter
from utils.load_dataset import ColumnAndTableRankerDataset

def parse_option():
    parser = argparse.ArgumentParser("command line arguments for fine-tuning pre-trained language model.")
    
    parser.add_argument('--batch_size', type = int, default = 8,
                        help = 'input batch size.')
    parser.add_argument('--gradient_descent_step', type = int, default = 4,
                        help = 'perform gradient descent per "gradient_descent_step" steps.')
    parser.add_argument('--device', type = str, default = "3",
                        help = 'the id of used GPU device.')
    parser.add_argument('--learning_rate',type = float, default = 3e-5,
                        help = 'learning rate.')
    parser.add_argument('--gamma', type = float, default = 1.0,
                        help = 'gamma parameter in the focal loss. Recommended gamma: [0.0-2.0].')
    parser.add_argument('--alpha', type = float, default = 1.0,
                        help = 'alpha parameter in the focal loss. Must between [0.0-1.0].')
    parser.add_argument('--weight_decay', type = float, default = 1e-3,
                        help = 'weight decay.')
    parser.add_argument('--epochs', type = int, default = 96,
                        help = 'training epochs.')
    parser.add_argument('--patience', type = int, default = 48,
                        help = 'patience step in early stopping. (-1 means no early stopping.)')
    parser.add_argument('--save_path', type = str, default = "models/ranker_model",
                        help = 'save path of best fine-tuned model on validation set.')
    parser.add_argument('--tensorboard_save_path', type = str, default = None,
                        help = 'save path of tensorboard log.')
    parser.add_argument('--train_dataset_path', type = str, default = "data/pre-processing/preprocessed_train_spider.json",
                        help = 'path of pre-processed training dataset.')
    parser.add_argument('--dev_dataset_path', type = str, default = "data/pre-processing/preprocessed_dev.json",
                        help = 'path of pre-processed development dataset.')
    parser.add_argument('--test_dataset_path', type = str, default = "data/pre-processing/preprocessed_dev.json",
                        help = 'path of pre-processed test dataset.')
    parser.add_argument('--output_dataset_path', type = str, default = "data/pre-processing/ranked_dataset.json",
                        help = 'path of the ranked dataset (used in eval mode).')
    parser.add_argument('--model_name_or_path', type = str, default = "plm_files/grappa_files/grappa_large",
                        help = 
                            '''
                            pre-trained model name. 
                            options:
                                plm_files/roberta_files/roberta-large,
                                plm_files/grappa_files/grappa_large
                            '''
                        )
    parser.add_argument('--use_contents', action='store_true',
                        help = 'whether to integrate db contents into input sequence')
    parser.add_argument('--use_original_name', action='store_true',
                        help = 'whether to use the original name of tables and columns')
    parser.add_argument('--mode', type = str, default = "train",
                        help='trian, eval or test.')
    
    opt = parser.parse_args()

    return opt
    
def prepare_batch_inputs_and_labels(batch, tokenizer):
    batch_size = len(batch)
    
    batch_questions = [data[0] for data in batch]
    
    batch_table_names = [data[1] for data in batch]
    batch_table_labels = [data[2] for data in batch]

    batch_column_infos = [data[3] for data in batch]
    batch_column_labels = [data[4] for data in batch]

    batch_fk_infos = [data[5] for data in batch]
    
    batch_input_tokens, batch_column_info_ids, batch_table_name_ids, batch_column_number_in_each_table = [], [], [], []
    for batch_id in range(batch_size):
        input_tokens = [batch_questions[batch_id]]

        table_names_in_one_db = batch_table_names[batch_id]
        column_infos_in_one_db = batch_column_infos[batch_id]

        batch_column_number_in_each_table.append([len(column_infos_in_one_table) for column_infos_in_one_table in column_infos_in_one_db])

        column_info_ids, table_name_ids = [], []
        
        for table_id, table_name in enumerate(table_names_in_one_db):
            input_tokens.append("|")
            input_tokens.append(table_name)
            table_name_ids.append(len(input_tokens) - 1)
            input_tokens.append(":")
            
            for column_info in column_infos_in_one_db[table_id]:
                input_tokens.append(column_info)
                column_info_ids.append(len(input_tokens) - 1)
                input_tokens.append(",")
            
            input_tokens = input_tokens[:-1]
        
        batch_input_tokens.append(input_tokens)
        batch_column_info_ids.append(column_info_ids)
        batch_table_name_ids.append(table_name_ids)

    tokenized_inputs = tokenizer(
        batch_input_tokens, 
        return_tensors="pt", 
        is_split_into_words = True, 
        padding = "max_length", # set to ``True'' will lead to slightly unstable in RoBERTa embedding stage
        max_length = 512,
        truncation = True
    )

    batch_aligned_question_ids, batch_aligned_column_info_ids, batch_aligned_table_name_ids = [], [], []
    batch_aligned_table_labels, batch_aligned_column_labels = [], []
    
    for batch_id in range(batch_size):
        word_ids = tokenized_inputs.word_ids(batch_index = batch_id)

        aligned_question_ids, aligned_table_name_ids, aligned_column_info_ids = [], [], []
        aligned_table_labels, aligned_column_labels = [], []

        for token_id, word_id in enumerate(word_ids):
            if word_id == 0:
                aligned_question_ids.append(token_id)

        for t_id, table_name_id in enumerate(batch_table_name_ids[batch_id]):
            temp_list = []
            for token_id, word_id in enumerate(word_ids):
                if table_name_id == word_id:
                    temp_list.append(token_id)
            if len(temp_list) != 0:
                aligned_table_name_ids.append(temp_list)
                aligned_table_labels.append(batch_table_labels[batch_id][t_id])

        for c_id, column_id in enumerate(batch_column_info_ids[batch_id]):
            temp_list = []
            for token_id, word_id in enumerate(word_ids):
                if column_id == word_id:
                    temp_list.append(token_id)
            if len(temp_list) != 0:
                aligned_column_info_ids.append(temp_list)
                aligned_column_labels.append(batch_column_labels[batch_id][c_id])

        batch_aligned_question_ids.append(aligned_question_ids)
        batch_aligned_table_name_ids.append(aligned_table_name_ids)
        batch_aligned_column_info_ids.append(aligned_column_info_ids)
        batch_aligned_table_labels.append(aligned_table_labels)
        batch_aligned_column_labels.append(aligned_column_labels)

    # remove truncated tables or columns
    for batch_id in range(batch_size):
        if len(batch_column_number_in_each_table[batch_id]) > len(batch_aligned_table_labels[batch_id]):
            batch_column_number_in_each_table[batch_id] = batch_column_number_in_each_table[batch_id][ : len(batch_aligned_table_labels[batch_id])]
        
        if sum(batch_column_number_in_each_table[batch_id]) > len(batch_aligned_column_labels[batch_id]):
            addtional_column_number = sum(batch_column_number_in_each_table[batch_id]) - len(batch_aligned_column_labels[batch_id])
            batch_column_number_in_each_table[batch_id][-1] -= addtional_column_number
    
    batch_fk_column_id_pairs = []
    for batch_id in range(batch_size):
        fk_column_id_pairs = []
        for fk_infos in batch_fk_infos[batch_id]:
            source_table_id, target_table_id, source_column_id, target_column_id = fk_infos
            if source_table_id + 1 > len(batch_column_number_in_each_table[batch_id]) or \
                target_table_id + 1 > len(batch_column_number_in_each_table[batch_id]):
                continue

            aligned_source_column_id = sum(batch_column_number_in_each_table[batch_id][0 : source_table_id]) + source_column_id
            aligned_target_column_id = sum(batch_column_number_in_each_table[batch_id][0 : target_table_id]) + target_column_id
            
            if aligned_source_column_id + 1 > sum(batch_column_number_in_each_table[batch_id]) or \
                aligned_target_column_id + 1 > sum(batch_column_number_in_each_table[batch_id]):
                continue

            fk_column_id_pairs.append([aligned_source_column_id, aligned_target_column_id])

        batch_fk_column_id_pairs.append(fk_column_id_pairs)

    # for batch_id in range(batch_size):
    #     print([token + " " + str(token_id) for token_id, token in enumerate(tokenizer.convert_ids_to_tokens(tokenized_inputs["input_ids"][batch_id]))])
    #     print(tokenizer.decode(tokenized_inputs["input_ids"][batch_id], skip_special_tokens = True))
    #     print("question_ids:", batch_aligned_question_ids[batch_id])
    #     print("table_name_ids:", batch_aligned_table_name_ids[batch_id])
    #     print("table_labels:", batch_aligned_table_labels[batch_id])
    #     print("column_info_ids:", batch_aligned_column_info_ids[batch_id])
    #     print("column_labels:", batch_aligned_column_labels[batch_id])
    #     print("column_number_in_each_table:", batch_column_number_in_each_table[batch_id])
    #     print("fk_column_id_pairs:", batch_fk_column_id_pairs[batch_id])
    #     print("fk_infos:", batch_fk_infos[batch_id])
    #     print("-------------------------------")

    encoder_input_ids = tokenized_inputs["input_ids"]
    encoder_input_attention_mask = tokenized_inputs["attention_mask"]
    batch_aligned_column_labels = [torch.LongTensor(column_labels) for column_labels in batch_aligned_column_labels]
    batch_aligned_table_labels = [torch.LongTensor(table_labels) for table_labels in batch_aligned_table_labels]

    if torch.cuda.is_available():
        encoder_input_ids = encoder_input_ids.cuda()
        encoder_input_attention_mask = encoder_input_attention_mask.cuda()
        batch_aligned_column_labels = [column_labels.cuda() for column_labels in batch_aligned_column_labels]
        batch_aligned_table_labels = [table_labels.cuda() for table_labels in batch_aligned_table_labels]

    # batch_fk_column_id_pairs, batch_column_infos, batch_questions, batch_table_names

    return encoder_input_ids, encoder_input_attention_mask, batch_aligned_column_labels, \
            batch_aligned_table_labels, batch_aligned_question_ids, \
            batch_aligned_column_info_ids, batch_aligned_table_name_ids, \
            batch_column_number_in_each_table

def splits(batch_column_labels, batch_column_number_in_each_table):
    batch_split_column_labels = []
    for batch_id in range(len(batch_column_labels)):
        split_column_labels = []
        for table_id in range(len(batch_column_number_in_each_table[batch_id])):
            split_column_labels.append(batch_column_labels[batch_id][sum(batch_column_number_in_each_table[batch_id][:table_id]) : sum(batch_column_number_in_each_table[batch_id][:table_id+1])])
        batch_split_column_labels.append(split_column_labels)
    
    return batch_split_column_labels

def rank_tables(batch_table_name_cls_logits):
    batch_table_ranking_indices = []

    for table_name_cls_logits in batch_table_name_cls_logits:
        with torch.no_grad():
            table_name_cls_probs = torch.nn.functional.softmax(table_name_cls_logits, dim = 1)

            probs = table_name_cls_probs[:, 1].cpu().numpy().tolist()
            probs = list(map(lambda x: round(x,4), probs))
            indices = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
            batch_table_ranking_indices.append(indices)
    
    return batch_table_ranking_indices

def rank_columns(
    batch_column_info_cls_logits, 
    batch_column_number_in_each_table
):
    batch_column_ranking_indices = []
    batch_size = len(batch_column_info_cls_logits)

    for batch_id in range(batch_size):
        column_number_in_each_table = batch_column_number_in_each_table[batch_id]

        with torch.no_grad():
            column_info_cls_probs = torch.nn.functional.softmax(batch_column_info_cls_logits[batch_id], dim = 1)
            
            column_ranking_indices_in_one_db = []
            for table_id in range(len(column_number_in_each_table)):
                column_info_cls_probs_in_one_table = column_info_cls_probs[sum(column_number_in_each_table[:table_id]) : sum(column_number_in_each_table[:table_id+1]), :]
                probs = column_info_cls_probs_in_one_table[:, 1].cpu().numpy().tolist()
                probs = list(map(lambda x: round(x,4), probs))
                indices = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
                column_ranking_indices_in_one_db.append(indices)

            batch_column_ranking_indices.append(column_ranking_indices_in_one_db)

    return batch_column_ranking_indices

def _train(opt):
    print(opt)
    set_seed(42)

    patience = opt.patience if opt.patience > 0 else float('inf')

    if opt.tensorboard_save_path is not None:
        writer = SummaryWriter(opt.tensorboard_save_path)
    else:
        writer = None

    os.environ["CUDA_VISIBLE_DEVICES"] = opt.device

    tokenizer = RobertaTokenizerFast.from_pretrained(
        opt.model_name_or_path,
        add_prefix_space = True
    )
    if "roberta" in opt.model_name_or_path:
        print("Let's use RoBERTa!")
    else:
        print("Let's use GRAPPA!")

    tokenizer.add_tokens(AddedToken("[FK]"))

    train_dataset = ColumnAndTableRankerDataset(
        dir_ = opt.train_dataset_path,
        use_original_name = opt.use_original_name,
        use_contents = opt.use_contents,
        use_column_type = False,
        add_pk_info = False,
        add_fk_info = True
    )

    train_dataloder = DataLoader(
        train_dataset, 
        batch_size = opt.batch_size, 
        shuffle = True,
        collate_fn = lambda x: x
    )

    dev_dataset = ColumnAndTableRankerDataset(
        dir_ = opt.dev_dataset_path,
        use_original_name = opt.use_original_name,
        use_contents = opt.use_contents,
        use_column_type = False,
        add_pk_info = False,
        add_fk_info = True
    )

    dev_dataloder = DataLoader(
        dev_dataset,
        batch_size = opt.batch_size,
        shuffle = False,
        collate_fn = lambda x: x
    )

    # Initialize model
    model = MyRanker(
        model_name_or_path = opt.model_name_or_path,
        vocab_size = len(tokenizer),
        mode = opt.mode
    )

    if torch.cuda.is_available():
        model = model.cuda()

    # warm up steps (10% training step)
    num_warmup_steps = int(0.1*opt.epochs*len(train_dataset)/opt.batch_size)
    # total training steps
    num_training_steps = int(opt.epochs*len(train_dataset)/opt.batch_size)
    # evaluate model for each 0.71425 training set (about 5000 examples)
    num_checkpoint_steps = int(0.71425 * len(train_dataset)/opt.batch_size)

    optimizer = optim.AdamW(
        params = model.parameters(), 
        lr = opt.learning_rate, 
        weight_decay = opt.weight_decay
    )

    # scheduler = transformers.get_constant_schedule_with_warmup(
    #     optimizer, 
    #     num_warmup_steps = num_warmup_steps,
    # )
    
    scheduler = transformers.get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps = num_warmup_steps,
        num_training_steps = num_training_steps
    )

    best_ranking_score = 0
    early_stop_step, train_step = 0, 0

    encoder_loss_func = RankerLoss(alpha = opt.alpha, gamma = opt.gamma)
    
    for epoch in range(opt.epochs):
        print(f"This is epoch {epoch+1}.")
        for batch in train_dataloder:
            model.train()
            train_step += 1

            encoder_input_ids, encoder_input_attention_mask, batch_column_labels, \
                batch_table_labels, batch_aligned_question_ids, \
                batch_aligned_column_info_ids, batch_aligned_table_name_ids, \
                batch_column_number_in_each_table = prepare_batch_inputs_and_labels(batch, tokenizer)
            
            model_outputs = model(
                encoder_input_ids,
                encoder_input_attention_mask,
                batch_aligned_question_ids, 
                batch_aligned_column_info_ids,
                batch_aligned_table_name_ids,
                batch_column_number_in_each_table
            )
            
            loss = encoder_loss_func.compute_loss(
                model_outputs["batch_table_name_cls_logits"],
                batch_table_labels,
                model_outputs["batch_column_info_cls_logits"],
                batch_column_labels
            )
            
            loss.backward()
            if scheduler is not None:
                scheduler.step()
            
            if writer is not None:
                writer.add_scalar('train loss', loss.item(), train_step)
                writer.add_scalar('train lr', optimizer.state_dict()['param_groups'][0]['lr'], train_step)
            
            if train_step % opt.gradient_descent_step == 0:
                optimizer.step()
                optimizer.zero_grad()
                
            if train_step % num_checkpoint_steps == 0:
                print(f"At {train_step} training step, start an evaluation.")
                model.eval()

                predict_column_labels, predict_table_labels = [], []
                ground_truth_column_labels, ground_truth_table_labels = [], []
                pred_table_ranking_indices, pred_column_ranking_indices = [], []
                gt_table_labels_for_table_ranking, gt_column_labels_for_column_ranking = [], []

                for batch in dev_dataloder:
                    encoder_input_ids, encoder_input_attention_mask, batch_column_labels, \
                        batch_table_labels, batch_aligned_question_ids, \
                        batch_aligned_column_info_ids, batch_aligned_table_name_ids, \
                        batch_column_number_in_each_table = prepare_batch_inputs_and_labels(batch, tokenizer)

                    with torch.no_grad():
                        model_outputs = model(
                            encoder_input_ids,
                            encoder_input_attention_mask,
                            batch_aligned_question_ids, 
                            batch_aligned_column_info_ids,
                            batch_aligned_table_name_ids,
                            batch_column_number_in_each_table
                        )

                    pred_table_ranking_indices += rank_tables(model_outputs["batch_table_name_cls_logits"])
                    pred_column_ranking_indices += rank_columns(
                        model_outputs["batch_column_info_cls_logits"], 
                        batch_column_number_in_each_table
                    )

                    batch_column_labels_list = [column_labels.detach().cpu().tolist() for column_labels in batch_column_labels]
                    batch_table_labels_list = [table_labels.detach().cpu().tolist() for table_labels in batch_table_labels]
                    
                    gt_table_labels_for_table_ranking += batch_table_labels_list
                    gt_column_labels_for_column_ranking += splits(batch_column_labels_list, batch_column_number_in_each_table)

                    for idx, table_name_cls_logits in enumerate(model_outputs["batch_table_name_cls_logits"]):
                        table_name_pred_labels = torch.argmax(table_name_cls_logits, dim = 1)
                        
                        predict_table_labels += table_name_pred_labels.cpu().numpy().tolist()
                        ground_truth_table_labels += batch_table_labels_list[idx]

                    for idx, column_info_cls_logits in enumerate(model_outputs["batch_column_info_cls_logits"]):
                        column_info_pred_labels = torch.argmax(column_info_cls_logits, dim = 1)
                        
                        predict_column_labels += column_info_pred_labels.cpu().numpy().tolist()
                        ground_truth_column_labels += batch_column_labels_list[idx]
                
                table_ranking_score = table_ranking_metric(pred_table_ranking_indices, gt_table_labels_for_table_ranking)
                column_ranking_score = column_ranking_metric(pred_column_ranking_indices, gt_column_labels_for_column_ranking)
                print("table_ranking_score:", table_ranking_score)
                print("column_ranking_score:", column_ranking_score)
                print("table+column ranking score:", table_ranking_score + column_ranking_score)

                table_cls_report = cls_metric(ground_truth_table_labels, predict_table_labels)
                column_cls_report = cls_metric(ground_truth_column_labels, predict_column_labels)
                print("table_cls_report:", table_cls_report)
                print("column_cls_report:", column_cls_report)

                if writer is not None:
                    writer.add_scalar("table ranking score", table_ranking_score, train_step/num_checkpoint_steps)
                    writer.add_scalar("column ranking score", column_ranking_score, train_step/num_checkpoint_steps)
                    writer.add_scalar("table+column ranking score", table_ranking_score + column_ranking_score, train_step/num_checkpoint_steps)
                    
                    writer.add_scalar('dev positive table/recall', table_cls_report["positives"]["recall"], train_step/num_checkpoint_steps)
                    writer.add_scalar('dev positive table/precision', table_cls_report["positives"]["precision"], train_step/num_checkpoint_steps)
                    writer.add_scalar('dev positive table/f1-score', table_cls_report["positives"]["f1-score"], train_step/num_checkpoint_steps)
                    writer.add_scalar('dev cls table acc', table_cls_report["accuracy"], train_step/num_checkpoint_steps)

                    writer.add_scalar('dev positive column/recall', column_cls_report["positives"]["recall"], train_step/num_checkpoint_steps)
                    writer.add_scalar('dev positive column/precision', column_cls_report["positives"]["precision"], train_step/num_checkpoint_steps)
                    writer.add_scalar('dev positive column/f1-score', column_cls_report["positives"]["f1-score"], train_step/num_checkpoint_steps)
                    writer.add_scalar('dev cls column acc', column_cls_report["accuracy"], train_step/num_checkpoint_steps)
                
                if epoch > 12:
                    return
                
                # save first few checkpoints for generating ranked training set for our seq2seq text2sql model.
                if epoch >= 3 and epoch <= 12:
                    os.makedirs(opt.save_path + "/checkpoints-{}".format(train_step), exist_ok = True)
                    torch.save(model.state_dict(), opt.save_path + "/checkpoints-{}".format(train_step) + "/dense_ranker.pt")
                    model.plm_encoder.config.save_pretrained(save_directory = opt.save_path + "/checkpoints-{}".format(train_step))
                    tokenizer.save_pretrained(save_directory = opt.save_path + "/checkpoints-{}".format(train_step))
                
                # if table_ranking_score + column_ranking_score >= best_ranking_score:
                #     os.makedirs(opt.save_path, exist_ok = True)
                #     best_ranking_score = table_ranking_score + column_ranking_score
                #     # save model checkpoints
                #     torch.save(model.state_dict(), opt.save_path + "/dense_ranker.pt")
                #     # save config of used PLM
                #     model.plm_encoder.config.save_pretrained(save_directory = opt.save_path)
                #     # save tokenizer for fast loading
                #     tokenizer.save_pretrained(save_directory = opt.save_path)
                #     early_stop_step = 0
                # else:
                #     early_stop_step += 1
                
                print("early_stop_step:", early_stop_step)

            if early_stop_step >= patience:
                break
        
        if early_stop_step >= patience:
            print("Ranker training process triggers early stopping.")
            break
    
    print("best ranking score:", best_ranking_score)

def _test(opt):
    set_seed(42)

    os.environ["CUDA_VISIBLE_DEVICES"] = opt.device

    tokenizer = RobertaTokenizerFast.from_pretrained(
        opt.save_path,
        add_prefix_space = True
    )
    
    dataset = ColumnAndTableRankerDataset(
        dir_ = opt.test_dataset_path,
        use_original_name = opt.use_original_name,
        use_contents = opt.use_contents,
        use_column_type = False,
        add_pk_info = False,
        add_fk_info = True
    )

    dataloder = DataLoader(
        dataset,
        batch_size = opt.batch_size,
        shuffle = False,
        collate_fn = lambda x: x
    )

    # Initialize model
    model = MyRanker(
        model_name_or_path = opt.save_path,
        vocab_size = len(tokenizer),
        mode = opt.mode
    )
    
    model.load_state_dict(torch.load(opt.save_path + "/dense_ranker.pt", map_location=torch.device('cpu')))
    if torch.cuda.is_available():
        model = model.cuda()
    
    model.eval()

    predict_column_labels, predict_table_labels = [], []
    ground_truth_column_labels, ground_truth_table_labels = [], []

    pred_table_ranking_indices, pred_column_ranking_indices = [], []
    gt_table_labels_for_table_ranking, gt_column_labels_for_column_ranking = [], []

    for batch in tqdm(dataloder):
        encoder_input_ids, encoder_input_attention_mask, batch_column_labels, \
            batch_table_labels, batch_aligned_question_ids, \
            batch_aligned_column_info_ids, batch_aligned_table_name_ids, \
            batch_column_number_in_each_table = prepare_batch_inputs_and_labels(batch, tokenizer)

        with torch.no_grad():
            model_outputs = model(
                encoder_input_ids,
                encoder_input_attention_mask,
                batch_aligned_question_ids, 
                batch_aligned_column_info_ids,
                batch_aligned_table_name_ids,
                batch_column_number_in_each_table
            )

        pred_table_ranking_indices += rank_tables(model_outputs["batch_table_name_cls_logits"])
        pred_column_ranking_indices += rank_columns(
            model_outputs["batch_column_info_cls_logits"], 
            batch_column_number_in_each_table
        )
        
        batch_column_labels_list = [column_labels.detach().cpu().tolist() for column_labels in batch_column_labels]
        batch_table_labels_list = [table_labels.detach().cpu().tolist() for table_labels in batch_table_labels]
                    
        gt_table_labels_for_table_ranking += batch_table_labels_list
        gt_column_labels_for_column_ranking += splits(batch_column_labels_list, batch_column_number_in_each_table)

        for idx, table_name_cls_logits in enumerate(model_outputs["batch_table_name_cls_logits"]):
            table_name_pred_labels = torch.argmax(table_name_cls_logits, dim = 1)
            
            predict_table_labels += table_name_pred_labels.cpu().numpy().tolist()
            ground_truth_table_labels += batch_table_labels_list[idx]

        for idx, column_info_cls_logits in enumerate(model_outputs["batch_column_info_cls_logits"]):
            column_info_pred_labels = torch.argmax(column_info_cls_logits, dim = 1)
            
            predict_column_labels += column_info_pred_labels.cpu().numpy().tolist()
            ground_truth_column_labels += batch_column_labels_list[idx]

    if opt.mode == "eval":
        table_ranking_score = table_ranking_metric(pred_table_ranking_indices, gt_table_labels_for_table_ranking)
        column_ranking_score = column_ranking_metric(pred_column_ranking_indices, gt_column_labels_for_column_ranking)

        print("table_ranking_score:", table_ranking_score)
        print("column_ranking_score:", column_ranking_score)
        print("table+column ranking score:", table_ranking_score + column_ranking_score)
        
        table_cls_report = cls_metric(ground_truth_table_labels, predict_table_labels)
        column_cls_report = cls_metric(ground_truth_column_labels, predict_column_labels)
        print("table_cls_report:", table_cls_report)
        print("column_cls_report:", column_cls_report)
        
    return pred_table_ranking_indices, pred_column_ranking_indices


def resorting_tables_in_dataset(table_ranking_results, input_file_name, output_file_name):
    with open(input_file_name, "r") as f:
        dataset = json.load(f)

    for data_id in range(len(dataset)):
        table_ranking_result = table_ranking_results[data_id]
        data = dataset[data_id]
        if len(data["db_schema"]) > len(table_ranking_result):
            truncated_table_num = len(table_ranking_result)
            for i in range(truncated_table_num, len(data["db_schema"])):
                table_ranking_result.append(i)
        
        dataset[data_id]["db_schema"] = [dataset[data_id]["db_schema"][idx] for idx in table_ranking_result]
        dataset[data_id]["table_labels"] = [dataset[data_id]["table_labels"][idx] for idx in table_ranking_result]
        dataset[data_id]["column_labels"] = [dataset[data_id]["column_labels"][idx] for idx in table_ranking_result]
    
    dataset_str = json.dumps(dataset, indent = 2)
    with open(output_file_name, "w") as f:
        f.write(dataset_str)

    return 

def resorting_columns_in_dataset(column_ranking_results, input_file_name, output_file_name):
    with open(input_file_name, "r") as f:
        dataset = json.load(f)

    for data_id in range(len(dataset)):
        column_ranking_results_in_one_db = column_ranking_results[data_id]
        data = dataset[data_id]
        for table_id, column_ranking_results_in_one_table in enumerate(column_ranking_results_in_one_db):
            original_column_num = len(data["db_schema"][table_id]["column_names_original"])
            current_column_num = len(column_ranking_results_in_one_table)
            if original_column_num > current_column_num:
                for i in range(current_column_num, original_column_num):
                    column_ranking_results_in_one_table.append(i)
            
            dataset[data_id]["db_schema"][table_id]["column_names_original"] = [dataset[data_id]["db_schema"][table_id]["column_names_original"][idx] for idx in column_ranking_results_in_one_table]
            dataset[data_id]["db_schema"][table_id]["column_names"] = [dataset[data_id]["db_schema"][table_id]["column_names"][idx] for idx in column_ranking_results_in_one_table]
            dataset[data_id]["db_schema"][table_id]["db_contents"] = [dataset[data_id]["db_schema"][table_id]["db_contents"][idx] for idx in column_ranking_results_in_one_table]
            dataset[data_id]["db_schema"][table_id]["column_types"] = [dataset[data_id]["db_schema"][table_id]["column_types"][idx] for idx in column_ranking_results_in_one_table]
            dataset[data_id]["column_labels"][table_id] = [dataset[data_id]["column_labels"][table_id][idx] for idx in column_ranking_results_in_one_table]
    
    dataset_str = json.dumps(dataset, indent = 2)
    with open(output_file_name, "w") as f:
        f.write(dataset_str)

    return 

if __name__ == "__main__":
    opt = parse_option()
    if opt.mode == "train":
        _train(opt)
    elif opt.mode in ["eval", "test"]:
        table_ranking_results, column_ranking_results = _test(opt)
        resorting_columns_in_dataset(
            column_ranking_results, 
            input_file_name = opt.test_dataset_path,
            output_file_name = "temp_dataset.json"
        )

        resorting_tables_in_dataset(
            table_ranking_results, 
            input_file_name = "temp_dataset.json",
            output_file_name = opt.output_dataset_path
        )

        os.remove("temp_dataset.json")