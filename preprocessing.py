import json
import argparse

from utils.bridge_content_encoder import get_database_matches, get_matched_entries
from sql_metadata import Parser
from tqdm import tqdm

def parse_option():
    parser = argparse.ArgumentParser("command line arguments for fine-tuning pre-trained language model.")
    
    parser.add_argument('--use_contents', action = 'store_true')
    parser.add_argument('--table_path', type = str, default = "./data/spider/tables.json")
    parser.add_argument('--mode', type = str, default = "train", 
                        help = '''
                            options:
                                1. train (know target SQLs and cls labels)
                                2. eval (same as train)
                                3. test (don't know target SQLs and thus don't know cls labels)
                            ''')
    parser.add_argument('--input_dataset_path', type = str, default = "./data/spider/train_spider.json", 
                        help = '''
                            options:
                                1. ./data/spider/train_spider.json
                                2. ./data/spider/train.json
                                3. ./data/spider/dev.json
                            ''')
    parser.add_argument('--output_dataset_path', type = str, default = "./data/pre-processing/preprocessed_dataset.json", 
                        help = "the filepath of preprocessed dataset.")
    parser.add_argument('--db_path', type = str, default = "./data/spider/database", 
                        help = "the filepath of database.")
            
    opt = parser.parse_args()

    return opt

def get_db_contents(question, table, column_names_original, db_id, db_path):
    all_matches = []
    for column in column_names_original:
        matched_contents = get_database_matches(question, table, column, db_path + "/{}/{}.sqlite".format(db_id, db_id))
        matched_contents = sorted(matched_contents)
        all_matches.append(matched_contents)
    
    return all_matches

def get_db_schemas(all_dbs):
    db_schemas = {}

    for db in all_dbs:
        table_names_original = db["table_names_original"]
        table_names = db["table_names"]
        all_column_types = db["column_types"]
        column_names_original = db["column_names_original"]
        column_names = db["column_names"]

        db_schemas[db["db_id"]] = {}
        
        primary_keys, foreign_keys = [], []
        for pk_column_idx in db["primary_keys"]:
            pk_table_name_original = table_names_original[column_names_original[pk_column_idx][0]]
            pk_column_name_original = column_names_original[pk_column_idx][1]
            
            primary_keys.append(
                {
                    "table_name": pk_table_name_original.lower(), 
                    "column_name": pk_column_name_original.lower()
                }
            )

        db_schemas[db["db_id"]]["pk"] = primary_keys

        for source_column_idx, target_column_idx in db["foreign_keys"]:
            fk_source_table_name_original = table_names_original[column_names_original[source_column_idx][0]]
            fk_source_column_name_original = column_names_original[source_column_idx][1]

            fk_target_table_name_original = table_names_original[column_names_original[target_column_idx][0]]
            fk_target_column_name_original = column_names_original[target_column_idx][1]
            
            foreign_keys.append(
                {
                    "source_table_name": fk_source_table_name_original.lower(),
                    "source_column_name": fk_source_column_name_original.lower(),
                    "target_table_name": fk_target_table_name_original.lower(),
                    "target_column_name": fk_target_column_name_original.lower(),
                }
            )
        
        db_schemas[db["db_id"]]["fk"] = foreign_keys

        db_schemas[db["db_id"]]["tables"] = {}
        for idx, table_name_original in enumerate(table_names_original):
            table_name_original = table_name_original.lower()
            table_name = table_names[idx]

            column_types_list = []
            column_names_original_list = []
            column_names_list = []
            
            for column_idx, (table_idx, column_name_original) in enumerate(column_names_original):
                column_name_original = column_name_original.lower()
                if idx == table_idx:
                    column_names_original_list.append(column_name_original)
                    column_names_list.append(column_names[column_idx][1])
                    column_types_list.append(all_column_types[column_idx])
            
            db_schemas[db["db_id"]]["tables"][table_name_original] = {
                "table_name": table_name, 
                "column_names": column_names_list, 
                "column_names_original": column_names_original_list,
                "column_types": column_types_list
            }

    return db_schemas

def sort_table_by_substring_matching(question_info):
    matched_scores = []

    table_number = len(question_info["db_schema"])
    for table_id in range(table_number):
        table_info_list = []
        table_info_list.append(question_info["db_schema"][table_id]["table_name"])

        for column_name, column_content in zip(question_info["db_schema"][table_id]["column_names"], question_info["db_schema"][table_id]["db_contents"]):
            table_info_list.append(column_name)
            table_info_list += column_content
        
        matched_score = []
        all_matched_string_and_score = get_matched_entries(question_info["question"], table_info_list, 0.6, 0.6)
        if all_matched_string_and_score is not None:
            for matched_string_and_score in all_matched_string_and_score:
                matched_score.append(matched_string_and_score[1][2])
        else:
            matched_score.append(0)
        
        matched_scores.append(max(matched_score))
    
    sorted_idx_ele_list = sorted(enumerate(matched_scores), key=lambda x:x[1], reverse=True)
    sorted_idx = [x[0] for x in sorted_idx_ele_list]
    
    question_info["db_schema"] = [question_info["db_schema"][idx] for idx in sorted_idx]
    question_info["table_labels"] = [question_info["table_labels"][idx] for idx in sorted_idx]
    question_info["column_labels"] = [question_info["column_labels"][idx] for idx in sorted_idx]

    return question_info

def main(dataset_path, output_dataset_path, table_path, use_contents, mode, db_path):
    with open(dataset_path) as f:
        dataset = json.load(f)

    with open(table_path) as f:
        all_dbs = json.load(f)
    
    db_schemas = get_db_schemas(all_dbs)
    
    preprocessed_dataset = []

    for data in tqdm(dataset):
        if data['query'] == 'SELECT T1.company_name FROM Third_Party_Companies AS T1 JOIN Maintenance_Contracts AS T2 ON T1.company_id  =  T2.maintenance_contract_company_id JOIN Ref_Company_Types AS T3 ON T1.company_type_code  =  T3.company_type_code ORDER BY T2.contract_end_date DESC LIMIT 1':
            data['query'] = 'SELECT T1.company_type FROM Third_Party_Companies AS T1 JOIN Maintenance_Contracts AS T2 ON T1.company_id  =  T2.maintenance_contract_company_id ORDER BY T2.contract_end_date DESC LIMIT 1'
            data['query_toks'] = ['SELECT', 'T1.company_type', 'FROM', 'Third_Party_Companies', 'AS', 'T1', 'JOIN', 'Maintenance_Contracts', 'AS', 'T2', 'ON', 'T1.company_id', '=', 'T2.maintenance_contract_company_id', 'ORDER', 'BY', 'T2.contract_end_date', 'DESC', 'LIMIT', '1']
            data['query_toks_no_value'] =  ['select', 't1', '.', 'company_type', 'from', 'third_party_companies', 'as', 't1', 'join', 'maintenance_contracts', 'as', 't2', 'on', 't1', '.', 'company_id', '=', 't2', '.', 'maintenance_contract_company_id', 'order', 'by', 't2', '.', 'contract_end_date', 'desc', 'limit', 'value']
            data['question'] = 'What is the type of the company who concluded its contracts most recently?'
            data['question_toks'] = ['What', 'is', 'the', 'type', 'of', 'the', 'company', 'who', 'concluded', 'its', 'contracts', 'most', 'recently', '?']
        if data['query'].startswith('SELECT T1.fname FROM student AS T1 JOIN lives_in AS T2 ON T1.stuid  =  T2.stuid WHERE T2.dormid IN'):
            data['query'] = data['query'].replace('IN (SELECT T2.dormid)', 'IN (SELECT T3.dormid)')
            index = data['query_toks'].index('(') + 2
            assert data['query_toks'][index] == 'T2.dormid'
            data['query_toks'][index] = 'T3.dormid'
            index = data['query_toks_no_value'].index('(') + 2
            assert data['query_toks_no_value'][index] == 't2'
            data['query_toks_no_value'][index] = 't3'
        
        sql = data["query"]
        question = data["question"]
        db_id = data["db_id"]
        
        parsed_sql = Parser(sql.replace("\"","'"))
        sql_tokens = [token.value.lower() for token in parsed_sql.tokens]
        
        if mode in ["train", "eval"]:
            used_tables = [table.lower() for table in parsed_sql.tables] # table_names_original
            try:
                used_columns = [column.lower() for column in parsed_sql.columns] # column_names_original
                if "language" in sql_tokens and "language" not in used_columns:
                    used_columns.append("language")
                if "result" in sql_tokens and "result" not in used_columns:
                    used_columns.append("result")
                if "location" in sql_tokens and "location" not in used_columns:
                    used_columns.append("location")
                if "share" in sql_tokens and "share" not in used_columns:
                    used_columns.append("share")
                if "type" in sql_tokens and "type" not in used_columns:
                    used_columns.append("type")
                if "year" in sql_tokens and "year" not in used_columns:
                    used_columns.append("year")
                used_columns = used_columns if len(used_columns) != 0 else ["*"]
            except:
                used_columns = ["*"]
        elif mode == "test":
            used_tables = []
            used_columns = []
        else:
            raise ValueError('mode must be in "train", "eval" or "test"')
        
        all_table_names = [k for k, _ in db_schemas[db_id]["tables"].items()]

        question_info = {}
        question_info["question"] = question
        question_info["query"] = sql
        question_info["db_id"] = db_id
        question_info["db_schema"] = []
        question_info["pk"] = db_schemas[db_id]["pk"]
        question_info["fk"] = db_schemas[db_id]["fk"]
        question_info["table_labels"] = []
        question_info["used_tables"] = used_tables
        question_info["column_labels"] = []
        question_info["used_columns"] = used_columns

        for table in all_table_names:
            column_names = db_schemas[db_id]["tables"][table]["column_names"]
            column_types = db_schemas[db_id]["tables"][table]["column_types"]
            # random.shuffle(column_names)
            column_names_original = db_schemas[db_id]["tables"][table]["column_names_original"]

            db_contents = get_db_contents(question, table, column_names_original, db_id, db_path) if use_contents else [[] for i in range(len(column_names))]

            column_names_original = column_names_original 
            column_names = column_names
            
            question_info["db_schema"].append(
                {
                    "table_name_original": table, 
                    "table_name": db_schemas[db_id]["tables"][table]["table_name"], 
                    "column_names_original": column_names_original,
                    "column_names": column_names, 
                    "db_contents": db_contents, 
                    "column_types": column_types 
                }
            )
            
            if mode in ["train", "eval"]:
                if table in used_tables:  # for used tables
                    question_info["table_labels"].append(1)
                    column_label = []
                    for column_name_original in column_names_original:
                        if column_name_original in used_columns or table+"."+column_name_original in used_columns:
                            column_label.append(1)
                        else:
                            column_label.append(0)
                    question_info["column_labels"].append(column_label)
                else:  # for unused tables
                    question_info["table_labels"].append(0)
                    question_info["column_labels"].append([0 for _ in range(len(column_names_original))])
            elif mode == "test":
                question_info["table_labels"].append(0)
                question_info["column_labels"].append([0 for _ in range(len(column_names_original))])

        preprocessed_dataset.append(question_info)

    with open(output_dataset_path, "w") as f:
        preprocessed_dataset_str = json.dumps(preprocessed_dataset, indent = 2)
        f.write(preprocessed_dataset_str)

if __name__ == "__main__":
    opt = parse_option()

    main(
        opt.input_dataset_path,
        opt.output_dataset_path, 
        opt.table_path,
        use_contents = opt.use_contents, 
        mode = opt.mode,
        db_path = opt.db_path
    )