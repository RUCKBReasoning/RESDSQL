import os
import json
import torch
import transformers
import argparse
import torch.optim as optim

from tqdm import tqdm
from copy import deepcopy
from tokenizers import AddedToken
from utils.classifier_metric.evaluator import cls_metric, auc_metric
from torch.utils.data import DataLoader
from transformers import RobertaTokenizerFast, XLMRobertaTokenizerFast
from utils.classifier_model import MyClassifier
from utils.classifier_loss import ClassifierLoss
from transformers.trainer_utils import set_seed
from torch.utils.tensorboard import SummaryWriter
from utils.load_dataset import ColumnAndTableClassifierDataset

def parse_option():
    parser = argparse.ArgumentParser("command line arguments for fine-tuning schema item classifier.")
    
    parser.add_argument('--batch_size', type = int, default = 8,
                        help = 'input batch size.')
    parser.add_argument('--gradient_descent_step', type = int, default = 4,
                        help = 'perform gradient descent per "gradient_descent_step" steps.')
    parser.add_argument('--device', type = str, default = "3",
                        help = 'the id of used GPU device.')
    parser.add_argument('--learning_rate',type = float, default = 3e-5,
                        help = 'learning rate.')
    parser.add_argument('--gamma', type = float, default = 1.0,
                        help = 'gamma parameter in the focal loss. Recommended: [0.0-2.0].')
    parser.add_argument('--alpha', type = float, default = 1.0,
                        help = 'alpha parameter in the focal loss. Must between [0.0-1.0].')
    parser.add_argument('--epochs', type = int, default = 128,
                        help = 'training epochs.')
    parser.add_argument('--patience', type = int, default = 32,
                        help = 'patience step in early stopping. -1 means no early stopping.')
    parser.add_argument('--seed', type = int, default = 42,
                        help = 'random seed.')
    parser.add_argument('--save_path', type = str, default = "models/schema_item_classifier",
                        help = 'save path of best fine-tuned model on validation set.')
    parser.add_argument('--tensorboard_save_path', type = str, default = None,
                        help = 'save path of tensorboard log.')
    parser.add_argument('--train_filepath', type = str, default = "data/pre-processing/preprocessed_train_spider.json",
                        help = 'path of pre-processed training dataset.')
    parser.add_argument('--dev_filepath', type = str, default = "data/pre-processing/preprocessed_dev.json",
                        help = 'path of pre-processed development dataset.')
    parser.add_argument('--output_filepath', type = str, default = "data/pre-processing/dataset_with_pred_probs.json",
                        help = 'path of the output dataset (used in eval mode).')
    parser.add_argument('--model_name_or_path', type = str, default = "roberta-large",
                        help = '''pre-trained model name.''')
    parser.add_argument('--use_contents', action='store_true',
                        help = 'whether to integrate db contents into input sequence')
    parser.add_argument('--add_fk_info', action='store_true',
                        help = 'whether to add [FK] tokens into input sequence')
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

    # notice: the trunction operation will discard some tables and columns that exceed the max length
    tokenized_inputs = tokenizer(
        batch_input_tokens, 
        return_tensors="pt", 
        is_split_into_words = True, 
        padding = "max_length",
        max_length = 512,
        truncation = True
    )

    batch_aligned_question_ids, batch_aligned_column_info_ids, batch_aligned_table_name_ids = [], [], []
    batch_aligned_table_labels, batch_aligned_column_labels = [], []
    
    # align batch_question_ids, batch_column_info_ids, and batch_table_name_ids after tokenizing
    for batch_id in range(batch_size):
        word_ids = tokenized_inputs.word_ids(batch_index = batch_id)

        aligned_question_ids, aligned_table_name_ids, aligned_column_info_ids = [], [], []
        aligned_table_labels, aligned_column_labels = [], []

        # align question tokens
        for token_id, word_id in enumerate(word_ids):
            if word_id == 0:
                aligned_question_ids.append(token_id)

        # align table names
        for t_id, table_name_id in enumerate(batch_table_name_ids[batch_id]):
            temp_list = []
            for token_id, word_id in enumerate(word_ids):
                if table_name_id == word_id:
                    temp_list.append(token_id)
            # if the tokenizer doesn't discard current table name
            if len(temp_list) != 0:
                aligned_table_name_ids.append(temp_list)
                aligned_table_labels.append(batch_table_labels[batch_id][t_id])

        # align column names
        for c_id, column_id in enumerate(batch_column_info_ids[batch_id]):
            temp_list = []
            for token_id, word_id in enumerate(word_ids):
                if column_id == word_id:
                    temp_list.append(token_id)
            # if the tokenizer doesn't discard current column name
            if len(temp_list) != 0:
                aligned_column_info_ids.append(temp_list)
                aligned_column_labels.append(batch_column_labels[batch_id][c_id])

        batch_aligned_question_ids.append(aligned_question_ids)
        batch_aligned_table_name_ids.append(aligned_table_name_ids)
        batch_aligned_column_info_ids.append(aligned_column_info_ids)
        batch_aligned_table_labels.append(aligned_table_labels)
        batch_aligned_column_labels.append(aligned_column_labels)

    # update column number in each table (because some tables and columns are discarded)
    for batch_id in range(batch_size):
        if len(batch_column_number_in_each_table[batch_id]) > len(batch_aligned_table_labels[batch_id]):
            batch_column_number_in_each_table[batch_id] = batch_column_number_in_each_table[batch_id][ : len(batch_aligned_table_labels[batch_id])]
        
        if sum(batch_column_number_in_each_table[batch_id]) > len(batch_aligned_column_labels[batch_id]):
            truncated_column_number = sum(batch_column_number_in_each_table[batch_id]) - len(batch_aligned_column_labels[batch_id])
            batch_column_number_in_each_table[batch_id][-1] -= truncated_column_number

    encoder_input_ids = tokenized_inputs["input_ids"]
    encoder_input_attention_mask = tokenized_inputs["attention_mask"]
    batch_aligned_column_labels = [torch.LongTensor(column_labels) for column_labels in batch_aligned_column_labels]
    batch_aligned_table_labels = [torch.LongTensor(table_labels) for table_labels in batch_aligned_table_labels]

    # print("\n".join(tokenizer.batch_decode(encoder_input_ids, skip_special_tokens = True)))

    if torch.cuda.is_available():
        encoder_input_ids = encoder_input_ids.cuda()
        encoder_input_attention_mask = encoder_input_attention_mask.cuda()
        batch_aligned_column_labels = [column_labels.cuda() for column_labels in batch_aligned_column_labels]
        batch_aligned_table_labels = [table_labels.cuda() for table_labels in batch_aligned_table_labels]

    return encoder_input_ids, encoder_input_attention_mask, \
        batch_aligned_column_labels, batch_aligned_table_labels, \
        batch_aligned_question_ids, batch_aligned_column_info_ids, \
        batch_aligned_table_name_ids, batch_column_number_in_each_table

def _train(opt):
    print(opt)
    set_seed(opt.seed)

    patience = opt.patience if opt.patience > 0 else float('inf')

    if opt.tensorboard_save_path is not None:
        writer = SummaryWriter(opt.tensorboard_save_path)
    else:
        writer = None

    os.environ["CUDA_VISIBLE_DEVICES"] = opt.device

    tokenizer_class = XLMRobertaTokenizerFast if "xlm" in opt.model_name_or_path else RobertaTokenizerFast

    tokenizer = tokenizer_class.from_pretrained(
        opt.model_name_or_path,
        add_prefix_space = True
    )
    tokenizer.add_tokens(AddedToken("[FK]"))

    train_dataset = ColumnAndTableClassifierDataset(
        dir_ = opt.train_filepath,
        use_contents = opt.use_contents,
        add_fk_info = opt.add_fk_info
    )

    train_dataloder = DataLoader(
        train_dataset, 
        batch_size = opt.batch_size, 
        shuffle = True,
        collate_fn = lambda x: x
    )

    dev_dataset = ColumnAndTableClassifierDataset(
        dir_ = opt.dev_filepath,
        use_contents = opt.use_contents,
        add_fk_info = opt.add_fk_info
    )

    dev_dataloder = DataLoader(
        dev_dataset,
        batch_size = opt.batch_size,
        shuffle = False,
        collate_fn = lambda x: x
    )

    # initialize model
    model = MyClassifier(
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
    # evaluate model for each 1.42857 epochs (about 1.42857*7000=10000 examples for Spider)
    num_checkpoint_steps = int(1.42857*len(train_dataset)/opt.batch_size)

    optimizer = optim.AdamW(
        params = model.parameters(), 
        lr = opt.learning_rate
    )

    scheduler = transformers.get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps = num_warmup_steps,
        num_training_steps = num_training_steps
    )

    best_score, early_stop_step, train_step = 0, 0, 0
    encoder_loss_func = ClassifierLoss(alpha = opt.alpha, gamma = opt.gamma)
    
    for epoch in range(opt.epochs):
        print(f"This is epoch {epoch+1}.")
        for batch in train_dataloder:
            model.train()
            train_step += 1

            encoder_input_ids, encoder_input_attention_mask, \
                batch_column_labels, batch_table_labels, batch_aligned_question_ids, \
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
            
            # update lr
            if scheduler is not None:
                scheduler.step()
            
            if writer is not None:
                # record training loss (tensorboard)
                writer.add_scalar('train loss', loss.item(), train_step)
                # record learning rate (tensorboard)
                writer.add_scalar('train lr', optimizer.state_dict()['param_groups'][0]['lr'], train_step)
            
            if train_step % opt.gradient_descent_step == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            if train_step % num_checkpoint_steps == 0:
                print(f"At {train_step} training step, start an evaluation.")
                model.eval()

                table_labels_for_auc, column_labels_for_auc = [], []
                table_pred_probs_for_auc, column_pred_probs_for_auc = [], []

                for batch in dev_dataloder:
                    encoder_input_ids, encoder_input_attention_mask, \
                        batch_column_labels, batch_table_labels, batch_aligned_question_ids, \
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

                    for batch_id, table_logits in enumerate(model_outputs["batch_table_name_cls_logits"]):
                        table_pred_probs = torch.nn.functional.softmax(table_logits, dim = 1)
                        
                        table_pred_probs_for_auc.extend(table_pred_probs[:, 1].cpu().tolist())
                        table_labels_for_auc.extend(batch_table_labels[batch_id].cpu().tolist())

                    for batch_id, column_logits in enumerate(model_outputs["batch_column_info_cls_logits"]):
                        column_pred_probs = torch.nn.functional.softmax(column_logits, dim = 1)
            
                        column_pred_probs_for_auc.extend(column_pred_probs[:, 1].cpu().tolist())
                        column_labels_for_auc.extend(batch_column_labels[batch_id].cpu().tolist())

                # calculate AUC score for table classification
                table_auc = auc_metric(table_labels_for_auc, table_pred_probs_for_auc)
                # calculate AUC score for column classification
                column_auc = auc_metric(column_labels_for_auc, column_pred_probs_for_auc)
                print("table AUC:", table_auc)
                print("column AUC:", column_auc)

                if writer is not None:
                    writer.add_scalar('table AUC', table_auc, train_step/num_checkpoint_steps)
                    writer.add_scalar('column AUC', column_auc, train_step/num_checkpoint_steps)
                
                toral_auc_score = table_auc + column_auc
                print("total auc:", toral_auc_score)
                # save the best ckpt
                if toral_auc_score >= best_score:
                    best_score = toral_auc_score
                    os.makedirs(opt.save_path, exist_ok = True)
                    torch.save(model.state_dict(), opt.save_path + "/dense_classifier.pt")
                    model.plm_encoder.config.save_pretrained(save_directory = opt.save_path)
                    tokenizer.save_pretrained(save_directory = opt.save_path)
                    early_stop_step = 0
                else:
                    early_stop_step += 1
                
                print("early_stop_step:", early_stop_step)

            if early_stop_step >= patience:
                break
        
        if early_stop_step >= patience:
            print("Classifier training process triggers early stopping.")
            break
    
    print("best auc score:", best_score)

def _test(opt):
    set_seed(opt.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.device

    tokenizer_class = XLMRobertaTokenizerFast if "xlm" in opt.model_name_or_path else RobertaTokenizerFast
    
    # load tokenizer
    tokenizer = tokenizer_class.from_pretrained(
        opt.save_path,
        add_prefix_space = True
    )
    
    dataset = ColumnAndTableClassifierDataset(
        dir_ = opt.dev_filepath,
        use_contents = opt.use_contents,
        add_fk_info = opt.add_fk_info
    )

    dataloder = DataLoader(
        dataset,
        batch_size = opt.batch_size,
        shuffle = False,
        collate_fn = lambda x: x
    )

    # initialize model
    model = MyClassifier(
        model_name_or_path = opt.save_path,
        vocab_size = len(tokenizer),
        mode = opt.mode
    )

    # load fine-tuned params
    model.load_state_dict(torch.load(opt.save_path + "/dense_classifier.pt", map_location=torch.device('cpu')))
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()

    table_labels_for_auc, column_labels_for_auc = [], []
    table_pred_probs_for_auc, column_pred_probs_for_auc = [], []

    returned_table_pred_probs, returned_column_pred_probs = [], []

    for batch in tqdm(dataloder):
        encoder_input_ids, encoder_input_attention_mask, \
            batch_column_labels, batch_table_labels, batch_aligned_question_ids, \
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
        
        for batch_id, table_logits in enumerate(model_outputs["batch_table_name_cls_logits"]):
            table_pred_probs = torch.nn.functional.softmax(table_logits, dim = 1)
            returned_table_pred_probs.append(table_pred_probs[:, 1].cpu().tolist())
            
            table_pred_probs_for_auc.extend(table_pred_probs[:, 1].cpu().tolist())
            table_labels_for_auc.extend(batch_table_labels[batch_id].cpu().tolist())

        for batch_id, column_logits in enumerate(model_outputs["batch_column_info_cls_logits"]):
            column_number_in_each_table = batch_column_number_in_each_table[batch_id]
            column_pred_probs = torch.nn.functional.softmax(column_logits, dim = 1)
            returned_column_pred_probs.append([column_pred_probs[:, 1].cpu().tolist()[sum(column_number_in_each_table[:table_id]):sum(column_number_in_each_table[:table_id+1])] \
                for table_id in range(len(column_number_in_each_table))])
            
            column_pred_probs_for_auc.extend(column_pred_probs[:, 1].cpu().tolist())
            column_labels_for_auc.extend(batch_column_labels[batch_id].cpu().tolist())

    if opt.mode == "eval":
        # calculate AUC score for table classification
        table_auc = auc_metric(table_labels_for_auc, table_pred_probs_for_auc)
        # calculate AUC score for column classification
        column_auc = auc_metric(column_labels_for_auc, column_pred_probs_for_auc)
        print("table auc:", table_auc)
        print("column auc:", column_auc)
        print("total auc:", table_auc+column_auc)
    
    return returned_table_pred_probs, returned_column_pred_probs

if __name__ == "__main__":
    opt = parse_option()
    if opt.mode == "train":
        _train(opt)
    elif opt.mode in ["eval", "test"]:
        total_table_pred_probs, total_column_pred_probs = _test(opt)
        
        with open(opt.dev_filepath, "r") as f:
            dataset = json.load(f)
        
        # record predicted probability
        truncated_data_info = []
        for data_id, data in enumerate(dataset):
            table_num = len(data["table_labels"])
            if table_num == len(total_table_pred_probs[data_id]):
                table_pred_probs = total_table_pred_probs[data_id]
            else:
                table_pred_probs = total_table_pred_probs[data_id] + [-1 for _ in range(table_num-len(total_table_pred_probs[data_id]))]
            
            truncated_table_ids = []
            column_pred_probs = []
            for table_id in range(table_num):
                if table_id >= len(total_column_pred_probs[data_id]):
                    truncated_table_ids.append(table_id)
                    column_pred_probs.append([-1 for _ in range(len(data["column_labels"][table_id]))])
                    continue
                if len(total_column_pred_probs[data_id][table_id]) == len(data["column_labels"][table_id]):
                    column_pred_probs.append(total_column_pred_probs[data_id][table_id])
                else:
                    truncated_table_ids.append(table_id)
                    truncated_column_num = len(data["column_labels"][table_id]) - len(total_column_pred_probs[data_id][table_id])
                    column_pred_probs.append(total_column_pred_probs[data_id][table_id] + [-1 for _ in range(truncated_column_num)])
            
            data["column_pred_probs"] = column_pred_probs
            data["table_pred_probs"] = table_pred_probs
            
            if len(truncated_table_ids) > 0:
                truncated_data_info.append([data_id, truncated_table_ids])

        # additionally, we need to consider and predict discarded tables and columns
        while len(truncated_data_info) != 0:
            truncated_dataset = []
            for truncated_data_id, truncated_table_ids in truncated_data_info:
                print(dataset[truncated_data_id]["question"])
                truncated_data = deepcopy(dataset[truncated_data_id])
                truncated_data["db_schema"] = [truncated_data["db_schema"][table_id] for table_id in truncated_table_ids]
                truncated_data["table_labels"] = [truncated_data["table_labels"][table_id] for table_id in truncated_table_ids]
                truncated_data["column_labels"] = [truncated_data["column_labels"][table_id] for table_id in truncated_table_ids]
                truncated_data["table_pred_probs"] = [truncated_data["table_pred_probs"][table_id] for table_id in truncated_table_ids]
                truncated_data["column_pred_probs"] = [truncated_data["column_pred_probs"][table_id] for table_id in truncated_table_ids]
                
                truncated_dataset.append(truncated_data)
            
            with open("./data/pre-processing/truncated_dataset.json", "w") as f:
                f.write(json.dumps(truncated_dataset, indent = 2, ensure_ascii = False))
            
            opt.dev_filepath = "./data/pre-processing/truncated_dataset.json"
            total_table_pred_probs, total_column_pred_probs = _test(opt)
            
            for data_id, data in enumerate(truncated_dataset):
                table_num = len(data["table_labels"])
                if table_num == len(total_table_pred_probs[data_id]):
                    table_pred_probs = total_table_pred_probs[data_id]
                else:
                    table_pred_probs = total_table_pred_probs[data_id] + [-1 for _ in range(table_num-len(total_table_pred_probs[data_id]))]

                column_pred_probs = []
                for table_id in range(table_num):
                    if table_id >= len(total_column_pred_probs[data_id]):
                        column_pred_probs.append([-1 for _ in range(len(data["column_labels"][table_id]))])
                        continue
                    if len(total_column_pred_probs[data_id][table_id]) == len(data["column_labels"][table_id]):
                        column_pred_probs.append(total_column_pred_probs[data_id][table_id])
                    else:
                        truncated_column_num = len(data["column_labels"][table_id]) - len(total_column_pred_probs[data_id][table_id])
                        column_pred_probs.append(total_column_pred_probs[data_id][table_id] + [-1 for _ in range(truncated_column_num)])
                
                # fill the predicted probability into the dataset
                truncated_data_id = truncated_data_info[data_id][0]
                truncated_table_ids = truncated_data_info[data_id][1]
                for idx, truncated_table_id in enumerate(truncated_table_ids):
                    dataset[truncated_data_id]["table_pred_probs"][truncated_table_id] = table_pred_probs[idx]
                    dataset[truncated_data_id]["column_pred_probs"][truncated_table_id] = column_pred_probs[idx]
            
            # check if there are tables and columns in the new dataset that have not yet been predicted
            truncated_data_info = []
            for data_id, data in enumerate(dataset):
                table_num = len(data["table_labels"])

                truncated_table_ids = []
                for table_id in range(table_num):
                    # the current table is not predicted
                    if data["table_pred_probs"][table_id] == -1:
                        truncated_table_ids.append(table_id)
                    # some columns in the current table are not predicted
                    if data["table_pred_probs"][table_id] != -1 and -1 in data["column_pred_probs"][table_id]:
                        truncated_table_ids.append(table_id)
                
                if len(truncated_table_ids) > 0:
                    truncated_data_info.append([data_id, truncated_table_ids])
            
            os.remove("./data/pre-processing/truncated_dataset.json")

        with open(opt.output_filepath, "w") as f:
            f.write(json.dumps(dataset, indent = 2, ensure_ascii = False))
