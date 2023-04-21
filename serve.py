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

app = FastAPI()
spider_db_dir = '/data/spider/database'
custom_db_dir = '/data/custom/database'

def create_sqlite_file(schema_json):
    sqlite_path = f'{custom_db_dir}/{schema_json["db_id"]}.sqlite'
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


class SQLiteSchema(BaseModel):
    db_schema: Dict


class UserQuery(BaseModel):
    db_id: Optional[str]
    query: str
    db_type: Literal['spider', 'custom'] = 'spider'
    # schema_json: Optional[str]

@app.get("/schema/spider/")
def schema_spider_list():
    '''list all spider schemas'''
    data = []
    sqlite_files = sorted([Path(f).stem for f in Path(spider_db_dir).rglob('*.sqlite')])
    return sqlite_files

@app.get("/schema/custom/")
def schema_spider_list():
    '''list all custom schemas'''
    data = []
    sqlite_files = sorted([Path(f).stem for f in Path(spider_db_dir).rglob('*.sqlite')])
    return sqlite_files

@app.get("/schema/custom/{db_id}")
def schema_spider_list(db_id):
    '''list all custom schemas'''
    db_path = f"{custom_db_dir}/{db_id}.sqlite"
    print(f'returning from {db_path}')
    return dump_db_json_schema(f'{custom_db_dir}/{db_id}.sqlite', db_id)

@app.post("/schema/custom/")
def schema_spider_create(schema: SQLiteSchema):
    create_sqlite_file(schema.db_schema)
    db_id = schema.db_schema['db_id']
    db_path = f"{custom_db_dir}/{db_id}.sqlite"
    return dump_db_json_schema(db_path, db_id)

@app.get("/schema/spider/{db_id}")
def schema_spider_get(db_id):
    data = dump_db_json_schema(f'{spider_db_dir}/{db_id}/{db_id}.sqlite', db_id)
    return data

@app.post("/resdsql/query")
def resdsql_query(user_query: UserQuery):
    # either db_id or scheme should be provided
    if user_query.db_id is None:
        raise HTTPException(status_code=400, detail="Either db_id or scheme should be provided")

    data = [{"db_id": user_query.db_id, "question": user_query.query, "query": ""}]

    temp_file_stem = f'{uuid.uuid4()}'
    Path(f'data/custom').mkdir(parents=True, exist_ok=True)
    with open(f'data/custom/{temp_file_stem}_input.json', 'w') as f:
        json.dump(data, f)
    result = os.system(f'sh scripts/inference/infer_text2sql.sh large custom {temp_file_stem}')
    print(f'inference script result : {result}')

    # read the result
    output = ""
    with open(f'predictions/{temp_file_stem}.sql', 'r') as f:
        output = f.read()
    return {"result": output}
