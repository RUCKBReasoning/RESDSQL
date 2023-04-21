import os
import sqlite3

from difflib import SequenceMatcher
from NatSQL.natsql_utils import natsql_to_sql
from func_timeout import func_set_timeout, FunctionTimedOut
from sql_metadata import Parser

def find_most_similar_sequence(source_sequence, target_sequences):
    max_match_length = -1
    most_similar_sequence = ""
    for target_sequence in target_sequences:
        match_length = SequenceMatcher(None, source_sequence, target_sequence).find_longest_match(0, len(source_sequence), 0, len(target_sequence)).size
        if max_match_length < match_length:
            max_match_length = match_length
            most_similar_sequence = target_sequence
    
    return most_similar_sequence

def tokenize_natsql(natsql):
    '''
        The function of tokenizing natsql.
        Two examples:
            Input1: select person.age from person where personfriend.name = 'Zach' and @.@ = max ( personfriend.year ) and personfriend.name = 'Zach'
            Output1: ['select', 'person.age', 'from', 'person', 'where', 'personfriend.name', '=', "'Zach'", 'and', '@.@', '=', 'max', '(', 'personfriend.year', ')', 'and', 'personfriend.name', '=', "'Zach'"]

            Input2: select sum ( order_items.order_quantity ) from customers where customers.customer_name = 'Rodrick Heaney'
            Output2: ['select', 'sum', '(', 'order_items.order_quantity', ')', 'from', 'customers', 'where', 'customers.customer_name', '=', "'Rodrick Heaney'"]
    '''
    # record all string values, e.g, 'Rodrick Heaney'
    in_string = False
    string_value_start_end_ids = []
    for c_id, c in enumerate(natsql):
        if c == "'":
            if in_string:
                string_value_start_end_ids.append(c_id)
                in_string = False
            else:
                string_value_start_end_ids.append(c_id)
                in_string = True
    string_values = []
    for start_id, end_id in zip(string_value_start_end_ids[0::2], string_value_start_end_ids[1::2]):
        string_values.append(natsql[start_id:end_id+1])
    
    # replace string values with a special placeholder 'string_value_placeholder'
    for string_value in set(string_values):
        natsql = natsql.replace(
            string_value, 
            "'string_value_placeholder'"
        )
    
    # tokenize by space char
    tokens = natsql.split()
    string_value_id = 0
    final_tokens = []
    # replace placeholders with real string values
    for token in tokens:
        if token == "'string_value_placeholder'":
            final_tokens.append(string_values[string_value_id])
            string_value_id += 1
        else:
            final_tokens.append(token)
    
    return final_tokens

def fix_fatal_errors_in_natsql(natsql, tc_original):
    '''
        Try to fix fatal schema item errors in the predicted natsql.
    '''
    tc_names = tc_original
    table_names = [tc_name.split(".")[0].strip() for tc_name in tc_names]
    column_names = [tc_name.split(".")[1].strip() for tc_name in tc_names]

    natsql_tokens = tokenize_natsql(natsql)
    new_tokens = []
    for idx, token in enumerate(natsql_tokens):
        # case A: current token is a wrong ``table.column'' name
        if "." in token and token != "@.@" and not token.startswith("'") and token not in tc_names:
            current_table_name = token.split(".")[0]
            current_column_name = token.split(".")[1]

            # case 1: both table name and column name are existing, but the column doesn't belong to the table
            if current_table_name in table_names and current_column_name in column_names:
                candidate_table_names = [table_name for table_name, column_name in zip(table_names, column_names) \
                    if current_column_name == column_name]
                new_table_name = find_most_similar_sequence(current_table_name, candidate_table_names)
                new_tokens.append(new_table_name+"."+current_column_name)
            # case 2: table name is not existing but column name is correct
            elif current_table_name not in table_names and current_column_name in column_names:
                candidate_table_names = [table_name for table_name, column_name in zip(table_names, column_names) \
                    if current_column_name == column_name]
                new_table_name = find_most_similar_sequence(current_table_name, candidate_table_names)
                new_tokens.append(new_table_name+"."+current_column_name)
            # case 3: table name is correct but column name is not existing
            elif current_table_name in table_names and current_column_name not in column_names:
                candidate_column_names = [column_name for table_name, column_name in zip(table_names, column_names) \
                    if current_table_name == table_name]
                new_column_name = find_most_similar_sequence(current_column_name, candidate_column_names)
                new_tokens.append(current_table_name+"."+new_column_name)
            # case 4: both table name and column name are not existing
            elif current_table_name not in table_names and current_column_name not in column_names:
                new_column_name = find_most_similar_sequence(current_column_name, column_names)
                candidate_table_names = [table_name for table_name, column_name in zip(table_names, column_names) \
                    if new_column_name == column_name]
                new_table_name = find_most_similar_sequence(current_table_name, candidate_table_names)
                new_tokens.append(new_table_name+"."+new_column_name)
        # case B: current token is a wrong ``table'' name
        elif natsql_tokens[idx-1] == "from" and token not in table_names:
            new_table_name = find_most_similar_sequence(token, list(set(table_names)))
            new_tokens.append(new_table_name)
        # case C: current token is right
        else:
            new_tokens.append(token)

    return " ".join(new_tokens)

# get the database cursor for a sqlite database path
def get_cursor_from_path(sqlite_path):
    try:
        if not os.path.exists(sqlite_path):
            print("Openning a new connection %s" % sqlite_path)
        connection = sqlite3.connect(sqlite_path, check_same_thread = False)
    except Exception as e:
        print(sqlite_path)
        raise e
    connection.text_factory = lambda b: b.decode(errors="ignore")
    cursor = connection.cursor()
    return cursor

# execute predicted sql with a time limitation
@func_set_timeout(120)
def execute_sql(cursor, sql):
    cursor.execute(sql)

    return cursor.fetchall()

def decode_natsqls(
    db_path,
    generator_outputs,
    batch_db_ids,
    batch_inputs,
    tokenizer,
    batch_tc_original,
    table_dict
):
    batch_size = generator_outputs.shape[0]
    num_return_sequences = generator_outputs.shape[1]

    final_sqls = []
    
    for batch_id in range(batch_size):
        pred_executable_sql = "sql placeholder"
        db_id = batch_db_ids[batch_id]
        db_file_path = db_path + "/{}/{}.sqlite".format(db_id, db_id)
        
        # print(batch_inputs[batch_id])
        # print("\n".join(tokenizer.batch_decode(generator_outputs[batch_id, :, :], skip_special_tokens = True)))

        for seq_id in range(num_return_sequences):
            cursor = get_cursor_from_path(db_file_path)
            pred_sequence = tokenizer.decode(generator_outputs[batch_id, seq_id, :], skip_special_tokens = True)

            pred_natsql = pred_sequence.split("|")[-1].strip()
            pred_natsql = pred_natsql.replace("='", "= '").replace("!=", " !=").replace(",", " ,")
            old_pred_natsql = pred_natsql
            # if the predicted natsql has some fatal errors, try to correct it
            pred_natsql = fix_fatal_errors_in_natsql(pred_natsql, batch_tc_original[batch_id])
            if old_pred_natsql != pred_natsql:
                print("Before fix:", old_pred_natsql)
                print("After fix:", pred_natsql)
                print("---------------")
            pred_sql = natsql_to_sql(pred_natsql, db_id, db_file_path, table_dict[db_id]).strip()
            
            # try to execute the predicted sql
            try:
                # Note: execute_sql will be success for empty string
                assert len(pred_sql) > 0, "pred sql is empty!"

                results = execute_sql(cursor, pred_sql)
                cursor.close()
                cursor.connection.close()
                # if the current sql has no execution error, we record and return it
                pred_executable_sql = pred_sql
                break
            except Exception as e:
                print(pred_sql)
                print(e)
                cursor.close()
                cursor.connection.close()
            except FunctionTimedOut as fto:
                print(pred_sql)
                print(fto)
                del cursor
        
        final_sqls.append(pred_executable_sql)
    
    return final_sqls

def decode_sqls(
    db_path,
    generator_outputs,
    batch_db_ids,
    batch_inputs,
    tokenizer,
    batch_tc_original
):
    batch_size = generator_outputs.shape[0]
    num_return_sequences = generator_outputs.shape[1]

    final_sqls = []
    
    for batch_id in range(batch_size):
        pred_executable_sql = "sql placeholder"
        db_id = batch_db_ids[batch_id]
        db_file_path = db_path + "/{}/{}.sqlite".format(db_id, db_id)
        
        # print(batch_inputs[batch_id])
        # print("\n".join(tokenizer.batch_decode(generator_outputs[batch_id, :, :], skip_special_tokens = True)))

        for seq_id in range(num_return_sequences):
            cursor = get_cursor_from_path(db_file_path)
            pred_sequence = tokenizer.decode(generator_outputs[batch_id, seq_id, :], skip_special_tokens = True)

            pred_sql = pred_sequence.split("|")[-1].strip()
            pred_sql = pred_sql.replace("='", "= '").replace("!=", " !=").replace(",", " ,")
            
            try:
                # Note: execute_sql will be success for empty string
                assert len(pred_sql) > 0, "pred sql is empty!"

                results = execute_sql(cursor, pred_sql)
                # if the current sql has no execution error, we record and return it
                pred_executable_sql = pred_sql
                cursor.close()
                cursor.connection.close()
                break
            except Exception as e:
                print(pred_sql)
                print(e)
                cursor.close()
                cursor.connection.close()
            except FunctionTimedOut as fto:
                print(pred_sql)
                print(fto)
                del cursor
        
        final_sqls.append(pred_executable_sql)
    
    return final_sqls
