import os
import torch
import sqlite3
import argparse
import torch.optim as optim

from tqdm import tqdm
from tokenizers import AddedToken
from torch.utils.data import DataLoader
from transformers import T5TokenizerFast, T5ForConditionalGeneration
import transformers
from transformers.optimization import Adafactor
from transformers.trainer_utils import set_seed
from torch.utils.tensorboard import SummaryWriter
from utils.spider_metric.evaluator import EvaluateTool
from utils.load_dataset import Text2SQLDataset

def parse_option():
    parser = argparse.ArgumentParser("command line arguments for fine-tuning pre-trained language model.")
    
    parser.add_argument('--batch_size', type = int, default = 8,
                        help = 'input batch size.')
    parser.add_argument('--gradient_descent_step', type = int, default = 4,
                        help = 'perform gradient descent per "gradient_descent_step" steps.')
    parser.add_argument('--device', type = str, default = "2",
                        help = 'the id of used GPU device.')
    parser.add_argument('--learning_rate',type = float, default = 3e-5,
                        help = 'learning rate.')
    parser.add_argument('--epochs', type = int, default = 96,
                        help = 'training epochs.')
    parser.add_argument('--patience', type = int, default = 32,
                        help = 'patience step in early stop.')
    parser.add_argument('--save_path', type = str, default = "models/text2sql1",
                        help = 'save path of best fine-tuned text2sql model.')
    parser.add_argument('--tensorboard_save_path', type = str, default = "tensorboard_log/text2sql1",
                        help = 'save path of tensorboard log.')
    parser.add_argument('--model_name_or_path', type = str, default = "plm_files/t5_models/t5.1.1.lm100k.large",
                        help = 
                        '''
                        pre-trained model name. 
                        options: plm_files/t5_models/t5.1.1.lm100k.base (tscholak/t5.1.1.lm100k.base, https://huggingface.co/tscholak/t5.1.1.lm100k.base)
                                 plm_files/t5_models/t5.1.1.lm100k.large (tscholak/t5.1.1.lm100k.large, https://huggingface.co/tscholak/t5.1.1.lm100k.large)
                                 plm_files/t5_models/t5.1.1.lm100k.xl (liangtaiwan/t5-v1_1-lm100k-xl, https://huggingface.co/liangtaiwan/t5-v1_1-lm100k-xl)
                        ''')
    parser.add_argument('--use_contents', action='store_true',
                        help = 'whether to integrate db contents into db schema')
    parser.add_argument('--use_adafactor', action='store_true',
                        help = 'whether to use adafactor to optimize model params.')
    parser.add_argument('--add_fk_info', action='store_true',
                        help = 'whether to add foreign key information into the db schema.')
    parser.add_argument('--add_pk_info', action='store_true',
                        help = 'whether to add primary key information into the db schema.')
    parser.add_argument('--output_sql_skeleton', action='store_true',
                        help = 'whether to add sql skeleton in the output sequence.')
    parser.add_argument('--mode', type = str, default = "train",
                        help='trian, eval or test.')
    parser.add_argument('--ranked_train_dataset_filepath', type = str, default = "data/pre-processing/ranked_train_spider7.17.json",
                        help = 'file path of upper bound training set.')
    parser.add_argument('--ranked_dev_dataset_filepath', type = str, default = "data/pre-processing/ranked_dev7.17.json",
                        help = 'file path of ranked dev set.')
    parser.add_argument('--ranked_test_dataset_filepath', type = str, default = "data/pre-processing/ranked_test7.17.json",
                        help = 'file path of ranked test set.')
    parser.add_argument('--original_dev_dataset_filepath', type = str, default = "data/spider/dev.json",
                        help = 'file path of original dev set. (for evaluation)')
    parser.add_argument('--db_path', type = str, default = "data/spider/database",
                        help = 'file path of database.')
    parser.add_argument('--checkpoint_type', type = str, default = "EM",
                        help = 'loading the best EXACT MATCH (EM) or EXECUTION (EXEC) checkpoint in the eval/test mode.')
    parser.add_argument('--num_beams', type = int, default = 8,
                        help = 'beam size in model.generate() function.')
    parser.add_argument('--num_return_sequences', type = int, default = 8,
                        help = 'the number of returned sequences in model.generate() function. (num_return_sequences <= num_beams)')

    opt = parser.parse_args()

    return opt

# get the database cursor for a sqlite database path
def get_cursor_from_path(sqlite_path: str):
    try:
        if not os.path.exists(sqlite_path):
            print("Openning a new connection %s" % sqlite_path)
        connection = sqlite3.connect(sqlite_path)
    except Exception as e:
        print(sqlite_path)
        raise e
    connection.text_factory = lambda b: b.decode(errors="ignore")
    cursor = connection.cursor()
    return cursor

def decode_sqls(
    opt,
    generator_outputs,
    batch_db_ids,
    batch_inputs,
    tokenizer
):
    '''
        generator_outputs: ids of decoder outputs, shape = batch_size x num_return_sequences x max_length
        tokenizer: object of T5Tokenizer
    '''
    batch_size = generator_outputs.shape[0]
    beam_size = generator_outputs.shape[1]

    final_sqls = []
    for batch_id in range(batch_size):
        db_id = batch_db_ids[batch_id]
        db_file_path = opt.db_path + "/{}/{}.sqlite".format(db_id, db_id)

        no_exec_error = False
        for beam_id in range(beam_size):
            cursor = get_cursor_from_path(db_file_path)
            pred_sql = tokenizer.decode(generator_outputs[batch_id, beam_id, :], skip_special_tokens = True)
            if opt.output_sql_skeleton and "|" in pred_sql:
                pred_sql = pred_sql.split("|")[-1].strip()
            try:
                cursor.execute(pred_sql)
                # result = cursor.fetchall()
                no_exec_error = True
                cursor.close()
                cursor.connection.close()
            except Exception as e:
                cursor.close()
                cursor.connection.close()
            if no_exec_error:
                final_sqls.append(pred_sql)
                break

        if no_exec_error is False:
            pred_sql = tokenizer.decode(generator_outputs[batch_id, 0, :], skip_special_tokens = True)
            if opt.output_sql_skeleton and "|" in pred_sql:
                pred_sql = pred_sql.split("|")[-1].strip()
            final_sqls.append(pred_sql)
    
    return final_sqls

def _train(opt):
    set_seed(42)
    print(opt)

    patience = opt.patience if opt.patience > 0 else float('inf')

    if opt.tensorboard_save_path is not None:
        writer = SummaryWriter(opt.tensorboard_save_path)
    else:
        writer = None

    os.environ["CUDA_VISIBLE_DEVICES"] = opt.device

    text2sql_tokenizer = T5TokenizerFast.from_pretrained(
        opt.model_name_or_path,
        add_prefix_space = True
    )
    
    if isinstance(text2sql_tokenizer, T5TokenizerFast):
        text2sql_tokenizer.add_tokens([AddedToken(" <="), AddedToken(" <")])
    
    train_dataset = Text2SQLDataset(
        dir_ = opt.ranked_train_dataset_filepath,
        use_contents = opt.use_contents,
        add_fk_info = opt.add_fk_info,
        add_pk_info = opt.add_pk_info,
        mode = "train",
        output_sql_skeleton = opt.output_sql_skeleton
    )

    train_dataloder = DataLoader(
        train_dataset, 
        batch_size = opt.batch_size, 
        shuffle = True,
        collate_fn = lambda x: x,
        drop_last = True
    )

    dev_dataset = Text2SQLDataset(
        dir_ = opt.ranked_dev_dataset_filepath,
        use_contents = opt.use_contents,
        add_fk_info = opt.add_fk_info,
        add_pk_info = opt.add_pk_info,
        mode = "eval",
        output_sql_skeleton = opt.output_sql_skeleton
    )

    dev_dataloder = DataLoader(
        dev_dataset, 
        batch_size = opt.batch_size, 
        shuffle = False,
        collate_fn = lambda x: x,
        drop_last = False
    )

    print("initializing text2sql model.")
    # initialize model
    text2sql_model = T5ForConditionalGeneration.from_pretrained(opt.model_name_or_path)
    text2sql_model.resize_token_embeddings(len(text2sql_tokenizer))
    if torch.cuda.is_available():
        text2sql_model = text2sql_model.cuda()
    
    print("finished.")

    # warm up steps (10% training step)
    num_warmup_steps = int(0.1*opt.epochs*len(train_dataset)/opt.batch_size)
    # total training steps
    num_training_steps = int(opt.epochs*len(train_dataset)/opt.batch_size)
    # evaluate model for each 0.71425 training set (about 5000 examples)
    num_checkpoint_steps = int(0.71425 * len(train_dataset)/opt.batch_size)

    # initialize evaluator
    evaluator = EvaluateTool()
    evaluator.register_golds(opt.original_dev_dataset_filepath, opt.db_path)

    if opt.use_adafactor:
        print("Let's use Adafactor as the optimizer!")
        optimizer = Adafactor(
            text2sql_model.parameters(), 
            lr=opt.learning_rate, 
            scale_parameter=False, 
            relative_step=False, 
            clip_threshold = 1.0,
            warmup_init=False
        )
    else:
        print("Let's use AdamW as the optimizer!")
        optimizer = optim.AdamW(
            text2sql_model.parameters(), 
            lr = opt.learning_rate
        )
    
    scheduler = transformers.get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps = num_warmup_steps,
        num_training_steps = num_training_steps
    )

    early_stop_step, train_step = 0, 0
    best_exact_match_score, best_execution_score = 0, 0
    best_em_metric_exec_score, best_exec_metric_exact_match_score = 0, 0

    for epoch in range(opt.epochs):
        print(f"This is epoch {epoch+1}.")
        for batch in train_dataloder:
            text2sql_model.train()
            train_step += 1
            
            batch_inputs = [data[0] for data in batch]
            batch_sqls = [data[1] for data in batch]
            batch_db_ids = [data[2] for data in batch]
            
            if epoch == 0:
                for batch_id in range(len(batch_inputs)):
                    print(batch_inputs[batch_id])
                    print(batch_sqls[batch_id])
                    print("----------------------")

            tokenized_inputs = text2sql_tokenizer(
                batch_inputs, 
                padding = "max_length",
                return_tensors = "pt",
                max_length = 512,
                truncation = True
            )
            
            with text2sql_tokenizer.as_target_tokenizer():
                tokenized_outputs = text2sql_tokenizer(
                    batch_sqls, 
                    padding = "max_length", 
                    return_tensors = 'pt',
                    max_length = 200,
                    truncation = True
                )
            
            encoder_input_ids = tokenized_inputs["input_ids"]
            encoder_input_attention_mask = tokenized_inputs["attention_mask"]

            decoder_labels = tokenized_outputs["input_ids"]
            decoder_labels[decoder_labels == text2sql_tokenizer.pad_token_id] = -100
            decoder_attention_mask = tokenized_outputs["attention_mask"]

            if torch.cuda.is_available():
                encoder_input_ids = encoder_input_ids.cuda()
                encoder_input_attention_mask = encoder_input_attention_mask.cuda()
                decoder_labels = decoder_labels.cuda()
                decoder_attention_mask = decoder_attention_mask.cuda()
            
            model_outputs = text2sql_model(
                input_ids = encoder_input_ids,
                attention_mask = encoder_input_attention_mask,
                labels = decoder_labels,
                decoder_attention_mask = decoder_attention_mask,
                return_dict = True
            )
            
            loss = model_outputs["loss"]
            loss.backward()

            if scheduler is not None:
                scheduler.step()

            if writer is not None:
                writer.add_scalar('train loss', loss.item(), train_step)
                writer.add_scalar('train lr', optimizer.state_dict()['param_groups'][0]['lr'], train_step)
            
            if train_step % opt.gradient_descent_step == 0:
                optimizer.step()
                optimizer.zero_grad()
                    
            if train_step % num_checkpoint_steps == 0 and epoch >= 8:
                print(f"At {train_step} training step, start an evaluation.")
                text2sql_model.eval()

                predict_sqls = []
                for batch in dev_dataloder:
                    batch_inputs = [data[0] for data in batch]
                    batch_db_ids = [data[1] for data in batch]
                    
                    if epoch == 0:
                        for batch_id in range(len(batch_inputs)):
                            print(batch_inputs[batch_id])
                            print("----------------------")
                    
                    tokenized_inputs = text2sql_tokenizer(
                        batch_inputs, 
                        return_tensors="pt",
                        padding = "max_length",
                        max_length = 512,
                        truncation = True
                    )
                    
                    encoder_input_ids = tokenized_inputs["input_ids"]
                    encoder_input_attention_mask = tokenized_inputs["attention_mask"]
                    if torch.cuda.is_available():
                        encoder_input_ids = encoder_input_ids.cuda()
                        encoder_input_attention_mask = encoder_input_attention_mask.cuda()

                    with torch.no_grad():
                        model_outputs = text2sql_model.generate(
                            input_ids = encoder_input_ids,
                            attention_mask = encoder_input_attention_mask,
                            max_length = 200,
                            decoder_start_token_id = text2sql_model.config.decoder_start_token_id,
                            num_beams = opt.num_beams,
                            num_return_sequences = opt.num_return_sequences
                        )

                    model_outputs = model_outputs.view(len(batch_inputs), opt.num_return_sequences, model_outputs.shape[1])
                    predict_sqls += decode_sqls(opt, model_outputs, batch_db_ids, batch_inputs, text2sql_tokenizer)
                
                spider_metric_result = evaluator.evaluate(predict_sqls)
                

                if writer is not None:
                    writer.add_scalar('spider dev set exact match', spider_metric_result["exact_match"], train_step/num_checkpoint_steps)
                    writer.add_scalar('spider dev set exec', spider_metric_result["exec"], train_step/num_checkpoint_steps)
                
                print("spider dev set exact match:", spider_metric_result["exact_match"])
                print("spider dev set exec:", spider_metric_result["exec"])

                os.makedirs(opt.save_path, exist_ok = True)
                if spider_metric_result["exact_match"] > best_exact_match_score or \
                    (spider_metric_result["exact_match"] == best_exact_match_score and spider_metric_result["exec"] > best_em_metric_exec_score):
                    best_exact_match_score = spider_metric_result["exact_match"]
                    best_em_metric_exec_score = spider_metric_result["exec"]
                    text2sql_model.save_pretrained(save_directory = opt.save_path + "/best_EM_model")
                    text2sql_tokenizer.save_pretrained(save_directory = opt.save_path + "/best_EM_model")
                    early_stop_step = 0
                else:
                    early_stop_step += 1
                
                if spider_metric_result["exec"] > best_execution_score or \
                    (spider_metric_result["exec"] == best_execution_score and spider_metric_result["exact_match"] > best_exec_metric_exact_match_score):
                    best_execution_score = spider_metric_result["exec"]
                    best_exec_metric_exact_match_score = spider_metric_result["exact_match"]
                    text2sql_model.save_pretrained(save_directory = opt.save_path + "/best_EXEC_model")
                    text2sql_tokenizer.save_pretrained(save_directory = opt.save_path + "/best_EXEC_model")
            
            if early_stop_step >= patience:
                break
        if early_stop_step >= patience:
            print("Text-to-SQL training process triggers early stopping.")
            break
    
    print("best exact match score:", best_exact_match_score)
    print("best execution score:", best_execution_score)
    
def _test(opt):
    set_seed(42)
    print(opt)

    import time
    start_time = time.time()

    if opt.checkpoint_type == "EM":
        model_save_path = opt.save_path + "/best_EM_model"
    elif opt.checkpoint_type == "EXEC":
        model_save_path = opt.save_path + "/best_EXEC_model"
    else:
        raise ValueError("Please select a correct type: EM or EXEC.")
    
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.device
    
    # initialize tokenizer
    tokenizer = T5TokenizerFast.from_pretrained(
        model_save_path,
        add_prefix_space = True
    )
    
    if isinstance(tokenizer, T5TokenizerFast):
        tokenizer.add_tokens([AddedToken(" <="), AddedToken(" <")])
    
    dev_dataset = Text2SQLDataset(
        dir_ = opt.ranked_test_dataset_filepath,
        use_contents = opt.use_contents,
        add_fk_info = opt.add_fk_info,
        add_pk_info = opt.add_pk_info,
        mode = opt.mode,
        output_sql_skeleton = opt.output_sql_skeleton
    )

    dev_dataloder = DataLoader(
        dev_dataset, 
        batch_size = opt.batch_size, 
        shuffle = False,
        collate_fn = lambda x: x,
        drop_last = False
    )

    # initialize model
    model = T5ForConditionalGeneration.from_pretrained(model_save_path)
    if torch.cuda.is_available():
        model = model.cuda()

    model.eval()
    predict_sqls = []
    for batch in tqdm(dev_dataloder):
        batch_inputs = [data[0] for data in batch]
        batch_db_ids = [data[1] for data in batch]

        tokenized_inputs = tokenizer(
            batch_inputs, 
            return_tensors="pt",
            padding = "max_length",
            max_length = 512,
            truncation = True
        )
        
        encoder_input_ids = tokenized_inputs["input_ids"]
        encoder_input_attention_mask = tokenized_inputs["attention_mask"]
        if torch.cuda.is_available():
            encoder_input_ids = encoder_input_ids.cuda()
            encoder_input_attention_mask = encoder_input_attention_mask.cuda()

        with torch.no_grad():
            model_outputs = model.generate(
                input_ids = encoder_input_ids,
                attention_mask = encoder_input_attention_mask,
                max_length = 200,
                decoder_start_token_id = model.config.decoder_start_token_id,
                num_beams = opt.num_beams,
                num_return_sequences = opt.num_return_sequences
            )

            model_outputs = model_outputs.view(len(batch_inputs), opt.num_return_sequences, model_outputs.shape[1])

        predict_sqls += decode_sqls(opt, model_outputs, batch_db_ids, batch_inputs, tokenizer)

    with open("predicted_sql.txt", "w", encoding = 'utf-8') as f:
        for pred in predict_sqls:
            f.write(pred + "\n")
    
    end_time = time.time()
    print("Text-to-SQL inference spends {}s.".format(end_time-start_time))
    
    if opt.mode == "eval":
        # initialize evaluator
        evaluator = EvaluateTool()
        evaluator.register_golds(opt.original_dev_dataset_filepath, opt.db_path)
        spider_metric_result = evaluator.evaluate(predict_sqls)
        print('exact_match score: {}'.format(spider_metric_result["exact_match"]))
        print('exec score: {}'.format(spider_metric_result["exec"]))
    
if __name__ == "__main__":
    opt = parse_option()
    if opt.mode in ["train"]:
        _train(opt)
    elif opt.mode in ["eval", "test"]:
        _test(opt)