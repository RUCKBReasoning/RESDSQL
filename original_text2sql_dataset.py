import json

input_dataset_path = "./data/preprocessed_data/preprocessed_train_spider.json"
output_dataset_path = "./data/preprocessed_data/resdsql_train_spider_wo_both.json"

# input_dataset_path = "./data/preprocessed_data/preprocessed_dev.json"
# output_dataset_path = "./data/preprocessed_data/resdsql_dev_wo_both.json"

dataset = json.load(open(input_dataset_path, "r"))
output_dataset = []

for data in dataset:
    db_id = data["db_id"]
    tc_original = []
    input_sequence = data["question"] + " | "
    for table in data["db_schema"]:
        input_sequence += table["table_name_original"] + " : "
        input_sequence += " , ".join(table["column_names_original"]) + " | "
        for column_name_original in table["column_names_original"]:
            tc_original.append(table["table_name_original"]+"."+column_name_original)

    for fk in data["fk"]:
        input_sequence += fk["source_table_name_original"]+"."+fk["source_column_name_original"]+" = "+fk["target_table_name_original"]+"."+fk["target_column_name_original"] + " | "
    
    output_sequence = data["norm_sql"]
    # output_sequence = data["sql_skeleton"] + " | " + data["norm_sql"]

    output_dataset.append({
        "db_id": db_id,
        "input_sequence": input_sequence,
        "output_sequence": output_sequence,
        "tc_original": tc_original
    })

with open(output_dataset_path, "w") as f:
    f.write(json.dumps(output_dataset, indent = 2, ensure_ascii = False))