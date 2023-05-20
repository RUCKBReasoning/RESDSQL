import json
import copy
import argparse
import random
import numpy as np

def parse_option():
    parser = argparse.ArgumentParser("command line arguments for generating the ranked dataset.")
    
    parser.add_argument('--input_dataset_path', type = str, default = "./data/pre-processing/dev_with_probs.json",
                        help = 'filepath of the input dataset.')
    parser.add_argument('--output_dataset_path', type = str, default = "./data/pre-processing/resdsql_dev.json",
                        help = 'filepath of the output dataset.')
    parser.add_argument('--topk_table_num', type = int, default = 4,
                        help = 'we only remain topk_table_num tables in the ranked dataset (k_1 in the paper).')
    parser.add_argument('--topk_column_num', type = int, default = 5,
                        help = 'we only remain topk_column_num columns for each table in the ranked dataset (k_2 in the paper).')
    parser.add_argument('--mode', type = str, default = "eval",
                        help = 'type of the input dataset, options: train, eval, test.')
    parser.add_argument('--noise_rate', type = float, default = 0.08,
                        help = 'the noise rate in the ranked training dataset (needed when the mode = "train")')
    parser.add_argument('--use_contents', action = 'store_true',
                        help = 'whether to add database contents in the input sequence.')
    parser.add_argument('--add_fk_info', action = 'store_true',
                    help = 'whether to add foreign key in the input sequence.')
    parser.add_argument('--output_skeleton', action = 'store_true',
                help = 'whether to add skeleton in the output sequence.')
    parser.add_argument("--target_type", type = str, default = "sql",
                help = "sql or natsql.")

    opt = parser.parse_args()

    return opt

def lista_contains_listb(lista, listb):
    for b in listb:
        if b not in lista:
            return 0
    
    return 1

def prepare_input_and_output(opt, ranked_data):
    question = ranked_data["question"]

    schema_sequence = ""
    for table_id in range(len(ranked_data["db_schema"])):
        table_name_original = ranked_data["db_schema"][table_id]["table_name_original"]
        # add table name
        schema_sequence += " | " + table_name_original + " : "
        
        column_info_list = []
        for column_id in range(len(ranked_data["db_schema"][table_id]["column_names_original"])):
            # extract column name
            column_name_original = ranked_data["db_schema"][table_id]["column_names_original"][column_id]
            db_contents = ranked_data["db_schema"][table_id]["db_contents"][column_id]
             # use database contents if opt.use_contents = True
            if opt.use_contents and len(db_contents) != 0:
                column_contents = " , ".join(db_contents)
                column_info = table_name_original + "." + column_name_original + " ( " + column_contents + " ) "
            else:
                column_info = table_name_original + "." + column_name_original

            column_info_list.append(column_info)
        
        if opt.target_type == "natsql":
            column_info_list.append(table_name_original + ".*")
    
        # add column names
        schema_sequence += " , ".join(column_info_list)

    if opt.add_fk_info:
        for fk in ranked_data["fk"]:
            schema_sequence += " | " + fk["source_table_name_original"] + "." + fk["source_column_name_original"] + \
                " = " + fk["target_table_name_original"] + "." + fk["target_column_name_original"]
    
    # remove additional spaces in the schema sequence
    while "  " in schema_sequence:
        schema_sequence = schema_sequence.replace("  ", " ")

    # input_sequence = question + schema sequence
    input_sequence = question + schema_sequence
        
    if opt.output_skeleton:
        if opt.target_type == "sql":
            output_sequence = ranked_data["sql_skeleton"] + " | " + ranked_data["norm_sql"]
        elif opt.target_type == "natsql":
            output_sequence = ranked_data["natsql_skeleton"] + " | " + ranked_data["norm_natsql"]
    else:
        if opt.target_type == "sql":
            output_sequence = ranked_data["norm_sql"]
        elif opt.target_type == "natsql":
            output_sequence = ranked_data["norm_natsql"]

    return input_sequence, output_sequence

def generate_train_ranked_dataset(opt):
    with open(opt.input_dataset_path) as f:
        dataset = json.load(f)
    
    output_dataset = []
    for data_id, data in enumerate(dataset):
        ranked_data = dict()
        ranked_data["question"] = data["question"]
        ranked_data["sql"] = data["sql"] # unused
        ranked_data["norm_sql"] = data["norm_sql"]
        ranked_data["sql_skeleton"] = data["sql_skeleton"]
        ranked_data["natsql"] = data["natsql"] # unused
        ranked_data["norm_natsql"] = data["norm_natsql"]
        ranked_data["natsql_skeleton"] = data["natsql_skeleton"]
        ranked_data["db_id"] = data["db_id"]
        ranked_data["db_schema"] = []

        # record ids of used tables
        used_table_ids = [idx for idx, label in enumerate(data["table_labels"]) if label == 1]
        topk_table_ids = copy.deepcopy(used_table_ids)

        if len(topk_table_ids) < opt.topk_table_num:
            remaining_table_ids = [idx for idx in range(len(data["table_labels"])) if idx not in topk_table_ids]
            # if topk_table_num is large than the total table number, all tables will be selected
            if opt.topk_table_num >= len(data["table_labels"]):
                topk_table_ids += remaining_table_ids
            # otherwise, we randomly select some unused tables
            else:
                randomly_sampled_table_ids = random.sample(remaining_table_ids, opt.topk_table_num - len(topk_table_ids))
                topk_table_ids += randomly_sampled_table_ids
        
        # add noise to the training set
        if random.random() < opt.noise_rate:
            random.shuffle(topk_table_ids)

        for table_id in topk_table_ids:
            new_table_info = dict()
            new_table_info["table_name_original"] = data["db_schema"][table_id]["table_name_original"]
            # record ids of used columns
            used_column_ids = [idx for idx, column_label in enumerate(data["column_labels"][table_id]) if column_label == 1]
            topk_column_ids = copy.deepcopy(used_column_ids)

            if len(topk_column_ids) < opt.topk_column_num:
                remaining_column_ids = [idx for idx in range(len(data["column_labels"][table_id])) if idx not in topk_column_ids]
                # same as the selection of top-k tables
                if opt.topk_column_num >= len(data["column_labels"][table_id]):
                    random.shuffle(remaining_column_ids)
                    topk_column_ids += remaining_column_ids
                else:
                    randomly_sampled_column_ids = random.sample(remaining_column_ids, opt.topk_column_num - len(topk_column_ids))
                    topk_column_ids += randomly_sampled_column_ids
            
            # add noise to the training set
            if random.random() < opt.noise_rate and table_id in used_table_ids:
                random.shuffle(topk_column_ids)
            
            new_table_info["column_names_original"] = [data["db_schema"][table_id]["column_names_original"][column_id] for column_id in topk_column_ids]
            new_table_info["db_contents"] = [data["db_schema"][table_id]["db_contents"][column_id] for column_id in topk_column_ids]
            
            ranked_data["db_schema"].append(new_table_info)

        # record foreign keys
        table_names_original = [table["table_name_original"] for table in data["db_schema"]]
        needed_fks = []
        for fk in data["fk"]:
            source_table_id = table_names_original.index(fk["source_table_name_original"])
            target_table_id = table_names_original.index(fk["target_table_name_original"])
            if source_table_id in topk_table_ids and target_table_id in topk_table_ids:
                needed_fks.append(fk)
        ranked_data["fk"] = needed_fks

        input_sequence, output_sequence = prepare_input_and_output(opt, ranked_data)
        
        # record table_name_original.column_name_original for subsequent correction function during inference
        tc_original = []
        for table in ranked_data["db_schema"]:
            for column_name_original in ["*"] + table["column_names_original"]:
                tc_original.append(table["table_name_original"] + "." + column_name_original)

        output_dataset.append(
            {
                "db_id": data["db_id"],
                "input_sequence": input_sequence, 
                "output_sequence": output_sequence,
                "tc_original": tc_original
            }
        )
    
    with open(opt.output_dataset_path, "w") as f:
        f.write(json.dumps(output_dataset, indent = 2, ensure_ascii = False))

def generate_eval_ranked_dataset(opt):
    with open(opt.input_dataset_path) as f:
        dataset = json.load(f)

    table_coverage_state_list, column_coverage_state_list = [], []
    output_dataset = []
    for data_id, data in enumerate(dataset):
        ranked_data = dict()
        ranked_data["question"] = data["question"]
        ranked_data["sql"] = data["sql"]
        ranked_data["norm_sql"] = data["norm_sql"]
        ranked_data["sql_skeleton"] = data["sql_skeleton"]
        ranked_data["natsql"] = data["natsql"]
        ranked_data["norm_natsql"] = data["norm_natsql"]
        ranked_data["natsql_skeleton"] = data["natsql_skeleton"]
        ranked_data["db_id"] = data["db_id"]
        ranked_data["db_schema"] = []

        table_pred_probs = list(map(lambda x:round(x,4), data["table_pred_probs"]))
        # find ids of tables that have top-k probability
        topk_table_ids = np.argsort(-np.array(table_pred_probs), kind="stable")[:opt.topk_table_num].tolist()
        
        # if the mode == eval, we record some information for calculating the coverage
        if opt.mode == "eval":
            used_table_ids = [idx for idx, label in enumerate(data["table_labels"]) if label == 1]
            table_coverage_state_list.append(lista_contains_listb(topk_table_ids, used_table_ids))
            
            for idx in range(len(data["db_schema"])):
                used_column_ids = [idx for idx, label in enumerate(data["column_labels"][idx]) if label == 1]
                if len(used_column_ids) == 0:
                    continue
                column_pred_probs = list(map(lambda x:round(x,2), data["column_pred_probs"][idx]))
                topk_column_ids = np.argsort(-np.array(column_pred_probs), kind="stable")[:opt.topk_column_num].tolist()
                column_coverage_state_list.append(lista_contains_listb(topk_column_ids, used_column_ids))

        # record top-k1 tables and top-k2 columns for each table
        for table_id in topk_table_ids:
            new_table_info = dict()
            new_table_info["table_name_original"] = data["db_schema"][table_id]["table_name_original"]
            column_pred_probs = list(map(lambda x:round(x,2), data["column_pred_probs"][table_id]))
            topk_column_ids = np.argsort(-np.array(column_pred_probs), kind="stable")[:opt.topk_column_num].tolist()
            
            new_table_info["column_names_original"] = [data["db_schema"][table_id]["column_names_original"][column_id] for column_id in topk_column_ids]
            new_table_info["db_contents"] = [data["db_schema"][table_id]["db_contents"][column_id] for column_id in topk_column_ids]
            
            ranked_data["db_schema"].append(new_table_info)
        
        # record foreign keys among selected tables
        table_names_original = [table["table_name_original"] for table in data["db_schema"]]
        needed_fks = []
        for fk in data["fk"]:
            source_table_id = table_names_original.index(fk["source_table_name_original"])
            target_table_id = table_names_original.index(fk["target_table_name_original"])
            if source_table_id in topk_table_ids and target_table_id in topk_table_ids:
                needed_fks.append(fk)
        ranked_data["fk"] = needed_fks
        
        input_sequence, output_sequence = prepare_input_and_output(opt, ranked_data)
        
        # record table_name_original.column_name_original for subsequent correction function during inference
        tc_original = []
        for table in ranked_data["db_schema"]:
            for column_name_original in table["column_names_original"] + ["*"]:
                tc_original.append(table["table_name_original"] + "." + column_name_original)

        output_dataset.append(
            {
                "db_id": data["db_id"],
                "input_sequence": input_sequence, 
                "output_sequence": output_sequence,
                "tc_original": tc_original
            }
        )
    
    with open(opt.output_dataset_path, "w") as f:
        f.write(json.dumps(output_dataset, indent = 2, ensure_ascii = False))
    
    if opt.mode == "eval":
        print("Table top-{} coverage: {}".format(opt.topk_table_num, sum(table_coverage_state_list)/len(table_coverage_state_list)))
        print("Column top-{} coverage: {}".format(opt.topk_column_num, sum(column_coverage_state_list)/len(column_coverage_state_list)))

if __name__ == "__main__":
    opt = parse_option()
    random.seed(42)
    
    if opt.mode == "train":
        generate_train_ranked_dataset(opt)
    elif opt.mode in ["eval", "test"]:
        generate_eval_ranked_dataset(opt)
    else:
        raise ValueError("The mode must be one of the ['train', 'eval', 'test'].")