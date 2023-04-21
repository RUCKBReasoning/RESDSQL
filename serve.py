from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Literal
import os
import json
import uuid
from pathlib import Path
from third_party.spider.preprocess.get_tables import dump_db_json_schema

app = FastAPI()
spider_db_dir = '/data/database'

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

@app.get("/schema/spider/{db_id}")
def schema_spider_get(db_id):
    data = dump_db_json_schema(f'/data/database/{db_id}/{db_id}.sqlite', db_id)
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
