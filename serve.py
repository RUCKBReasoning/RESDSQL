from fastapi import FastAPI
from pydantic import BaseModel
import os
import json

app = FastAPI()

class UserQuery(BaseModel):
    db_id: str
    query: str

@app.post("/resdsql/query")
def resdsql_query(user_query: UserQuery):
    data = [{"db_id": user_query.db_id, "question": user_query.query, "query": ""}]
    with open('data/spider/test.json', 'w') as f:
        json.dump(data, f)
    result = os.system('sh scripts/inference/infer_text2sql.sh large spider')
    print(f'inference script result : {result}')

    # read the result
    output = ""
    with open('predictions/Spider-dev/resdsql_large/pred.sql', 'r') as f:
        output = f.read()
    return {"result": output}
