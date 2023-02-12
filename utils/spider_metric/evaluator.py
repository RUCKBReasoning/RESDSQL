# encoding=utf8
import json
import os
from third_party.spider.preprocess.get_tables import dump_db_json_schema
from .spider_exact_match import compute_exact_match_metric
from .spider_test_suite import compute_test_suite_metric

class EvaluateTool(object):
    def __init__(self):
        # self.args = args
        self.schema_cache = dict()
        self.golds = []

    def register_golds(self, dataset_filepath, db_path):
        with open(dataset_filepath, encoding="utf-8") as f:
            dataset = json.load(f)
            for idx, sample in enumerate(dataset):
                if sample['query'] == 'SELECT T1.company_name FROM Third_Party_Companies AS T1 JOIN Maintenance_Contracts AS T2 ON T1.company_id  =  T2.maintenance_contract_company_id JOIN Ref_Company_Types AS T3 ON T1.company_type_code  =  T3.company_type_code ORDER BY T2.contract_end_date DESC LIMIT 1':
                    sample['query'] = 'SELECT T1.company_type FROM Third_Party_Companies AS T1 JOIN Maintenance_Contracts AS T2 ON T1.company_id  =  T2.maintenance_contract_company_id ORDER BY T2.contract_end_date DESC LIMIT 1'
                    sample['query_toks'] = ['SELECT', 'T1.company_type', 'FROM', 'Third_Party_Companies', 'AS', 'T1', 'JOIN', 'Maintenance_Contracts', 'AS', 'T2', 'ON', 'T1.company_id', '=', 'T2.maintenance_contract_company_id', 'ORDER', 'BY', 'T2.contract_end_date', 'DESC', 'LIMIT', '1']
                    sample['query_toks_no_value'] =  ['select', 't1', '.', 'company_type', 'from', 'third_party_companies', 'as', 't1', 'join', 'maintenance_contracts', 'as', 't2', 'on', 't1', '.', 'company_id', '=', 't2', '.', 'maintenance_contract_company_id', 'order', 'by', 't2', '.', 'contract_end_date', 'desc', 'limit', 'value']
                    sample['question'] = 'What is the type of the company who concluded its contracts most recently?'
                    sample['question_toks'] = ['What', 'is', 'the', 'type', 'of', 'the', 'company', 'who', 'concluded', 'its', 'contracts', 'most', 'recently', '?']
                if sample['query'].startswith('SELECT T1.fname FROM student AS T1 JOIN lives_in AS T2 ON T1.stuid  =  T2.stuid WHERE T2.dormid IN'):
                    sample['query'] = sample['query'].replace('IN (SELECT T2.dormid)', 'IN (SELECT T3.dormid)')
                    index = sample['query_toks'].index('(') + 2
                    assert sample['query_toks'][index] == 'T2.dormid'
                    sample['query_toks'][index] = 'T3.dormid'
                    index = sample['query_toks_no_value'].index('(') + 2
                    assert sample['query_toks_no_value'][index] == 't2'
                    sample['query_toks_no_value'][index] = 't3'
    
                db_id = sample["db_id"]
                if db_id not in self.schema_cache:
                    self.schema_cache[db_id] = dump_db_json_schema(
                        db=os.path.join(db_path, db_id, f"{db_id}.sqlite"), f=db_id
                    )
                schema = self.schema_cache[db_id]

                self.golds.append({
                    "query": sample["query"],
                    "question": sample["question"],
                    "db_id": db_id,
                    "db_path": db_path,
                    "db_table_names": schema["table_names_original"],
                    "db_column_names": {
                        "table_id": [table_id for table_id, _ in schema["column_names_original"]],
                        "column_name": [column_name for _, column_name in schema["column_names_original"]]
                    },
                    "db_column_types": schema["column_types"],
                    "db_primary_keys": [{"column_id": column_id} for column_id in schema["primary_keys"]],
                    "db_foreign_keys": {
                        "column_id": [column_id for column_id, _ in schema["foreign_keys"]],
                        "other_column_id": [other_column_id for _, other_column_id in schema["foreign_keys"]]
                    },
                })

    def evaluate(self, preds):
        exact_match = compute_exact_match_metric(preds, self.golds)
        test_suite = compute_test_suite_metric(preds, self.golds, db_dir = None)
        
        return {**exact_match, **test_suite}
