from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Literal, Dict
import os
import json
from collections import defaultdict
import uuid
from pathlib import Path
from third_party.spider.preprocess.get_tables import dump_db_json_schema
import sqlite3
import logging

logging.config.fileConfig('logging.conf', disable_existing_loggers=False)
logger = logging.getLogger(__name__) 

app = FastAPI()
spider_db_dir = '/data/spider/database'
query_dir = '/data/query'
sqlite_db_dict = {Path(f).stem:Path(f) for f in Path(spider_db_dir).rglob('*.sqlite')}
spider_db_ids = [o['db_id'] for o in json.load(open('/data/spider/tables.json'))]
Path(query_dir).mkdir(parents=True, exist_ok=True)

def get_sqlite_path(db_id, db_type):
    return f'{spider_db_dir}/{db_id}/{db_id}.sqlite'

def get_db_dict(force_refresh=False):
    global sqlite_db_dict
    if force_refresh:
        sqlite_db_dict = {Path(f).stem:Path(f) for f in Path(spider_db_dir).rglob('*.sqlite')}
    return sqlite_db_dict

def create_sqlite_file(schema_json):
    db_id = schema_json['db_id']
    if db_id in spider_db_ids:
        raise HTTPException(status_code=400, detail=f"db_id {db_id} already exists in spider")

    sqlite_path = get_sqlite_path(schema_json['db_id'], 'custom')
    logger.info(f'sqlite path is {sqlite_path}')
    # create parent if not exists
    Path(sqlite_path).parent.mkdir(parents=True, exist_ok=True)
    print(f'creating sqlite file at {sqlite_path}')
    sqlite_conn = sqlite3.connect(sqlite_path)
    cursor = sqlite_conn.cursor()

    tables = schema_json['table_names_original']
    columns = defaultdict(list)
    is_primary_list = []

    for col_idx, (table_idx, column_name) in enumerate(schema_json['column_names_original']):
        columns[table_idx].append(column_name)
        
        is_primary = False
        if 'primary_keys' in schema_json and col_idx in  schema_json['primary_keys']:
            is_primary = True
        is_primary_list.append(True)

        
    for tab_idx, table_name in enumerate(tables):
        
        cols_str = ', '.join([f"{col_name} {col_type}" for (table_idx,col_name), col_type in zip(schema_json['column_names_original'],schema_json['column_types']) if table_idx == tab_idx])
        create_table_query = f"create table {table_name} ({cols_str})"
        print(create_table_query)
        sqlite_conn.execute(create_table_query)
        
    sqlite_conn.commit()
    cursor.close()
    sqlite_conn.close()
    get_db_dict(force_refresh=True)
    return sqlite_path

class SQLiteSchema(BaseModel):
    db_schema: Dict


class UserQuery(BaseModel):
    db_id: Optional[str]
    query: str

def get_schema_json_path(db_id, temp_query_file_stem):
    if db_id in spider_db_ids:
        return f'{spider_db_dir}/{db_id}/tables.json'

    # custom sql db
    schema_json = dump_db_json_schema(f'{spider_db_dir}/{db_id}/{db_id}.sqlite', db_id)

    # save the schema json
    schema_json_path = f'{query_dir}/{temp_query_file_stem}_schema.json'
    with open(schema_json_path, 'w') as f:
        json.dump([schema_json], f)
    return schema_json_path

@app.get("/schema/spider/")
def schema_spider_list():
    '''list all spider schemas'''
    return spider_db_ids

@app.get("/schema/custom/")
def schema_custom_list():
    '''list all custom schemas'''
    return [db_id for db_id in get_db_dict().keys() if db_id not in spider_db_ids]

@app.get("/schema/custom/{db_id}")
def schema_spider_list(db_id):
    return dump_db_json_schema(f'{custom_db_dir}/{db_id}.sqlite', db_id)

@app.post("/schema/custom/")
def schema_custom_create(schema: SQLiteSchema):
    sqlite_path = create_sqlite_file(schema.db_schema)
    db_id = schema.db_schema['db_id']
    return dump_db_json_schema(sqlite_path, db_id)

@app.get("/schema/spider/{db_id}")
def schema_spider_get(db_id):
    data = dump_db_json_schema(f'{spider_db_dir}/{db_id}/{db_id}.sqlite', db_id)
    return data

@app.post("/resdsql/query")
def resdsql_query(user_query: UserQuery):
    if user_query.db_id is None:
        raise HTTPException(status_code=400, detail="db_id is required")
    
    if user_query.db_id not in get_db_dict():
        raise HTTPException(status_code=400, detail=f"db_id {user_query.db_id} does not exist")

    data = [{"db_id": user_query.db_id, "question": user_query.query, "query": ""}]

    db_type = 'spider'
    if user_query.db_id not in spider_db_ids:
        db_type = 'custom'


    temp_query_file_stem = f'{uuid.uuid4()}'
    input_path = f'{query_dir}/{temp_query_file_stem}_input.json'
    # write input to file
    with open(f'{input_path}', 'w') as f:
        json.dump(data, f)
    # write schema to file
    schema_json_path = get_schema_json_path(user_query.db_id, temp_query_file_stem)

    logger.info(f'input_path is {input_path} and schema json path is {schema_json_path}')
    result = os.system(f'sh scripts/inference/infer_text2sql.sh large {db_type} api {temp_query_file_stem}')
    print(f'inference script result : {result}')

    # read the result
    output = ""
    output_path = f'/data/query/{temp_query_file_stem}_output.sql'
    if not os.path.exists(output_path):
        raise HTTPException(status_code=400, detail=f"output file {output_path} not found")

    with open(output_path, 'r') as f:
        output = f.read()
    return {"result": output}
