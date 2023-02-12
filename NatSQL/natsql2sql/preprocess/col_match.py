from .table_match import return_column_match
from .utils import str_is_date

def col_match_not_in_table(match,table_idxs,schema):
    if match:
        for c in match[0]:
            if schema.column_names_original[c][0] in table_idxs:
                return False
    return True

def of_for_structure_in_col(col_availble_idx,q_toks):
    if len(col_availble_idx) != 3:
        return False
    off_1 = col_availble_idx[1][0] - col_availble_idx[0][-1] - 1
    off_2 = col_availble_idx[2][0] - col_availble_idx[1][-1] - 1

    if off_1 == 1 and off_2 == 1:
        if q_toks[col_availble_idx[1][0]-1].text == 'of':
            return True
    if off_1 == 2 and off_2 == 1:
        if q_toks[col_availble_idx[1][0]-1].text == 'the' and q_toks[col_availble_idx[1][0]-2].text == 'of':
            return True
    if off_1 == 2 and off_2 == 2:
        if q_toks[col_availble_idx[1][0]-1].text == 'the' and q_toks[col_availble_idx[1][0]-2].text == 'of':
            return True
    if off_1 == 1 and off_2 == 2:
        if q_toks[col_availble_idx[1][0]-1].text == 'of':
            return True
    return False




def col_match_main(tables,ts,schema,table_match,select_type=False,all_tables=None):
    """
        different from the same function in sentence_analyse.py
    """
    def col_match(tables,ts,schema,table_match,input_match=None,only_exact_match=False,is_not_all_tables=True,select_type=False):
        def special_case_for_pass(col_id,schema,ts,ts_id,select_type=False):
            for col in schema.column_tokens_lemma_str[col_id].split(" | "):
                if col.endswith(" name") or col.endswith(" id") or col.endswith(" type") or col.endswith(" code") or col.endswith(" value") or col.endswith(" count") or col.startswith("count ") or col.startswith("number ") or col.endswith(" number") or col.startswith("amount ") or col.endswith(" amount") or col.endswith(ts.tokens[ts_id].lemma_):
                    if select_type:
                        return True
                elif col.endswith(" rate") and (" 0." in ts.text or "%" in ts.text):
                    return True
                elif schema.column_types[col_id] == "time" and (" oldest " in ts.text or " latest " in ts.text or " earliest " in ts.text or " longest " in ts.text and " shortest " in ts.text or " after 1" in ts.text or " after 2" in ts.text or " before 1" in ts.text or " before 2" in ts.text or "when " in ts.text):
                    return True
                elif schema.column_types[col_id] == "time" and ts_id + 5 < len(ts.tokens) and (ts.tokens[ts_id+1].text == "between" or ts.tokens[ts_id+2].text == "between"):
                    date_count = 0
                    for i in range(2,6):
                        if str_is_date(ts.tokens[ts_id+i].text,ts.tokens,ts_id+i):
                           date_count += 1
                    if date_count >= 2:
                        return True 

            return False
        if input_match:
            col_matchs = [input_match]
        else:
            col_matchs = []
        for t in tables:
            _,full_match = return_column_match(ts,schema,t,None,only_exact_match)
            col_matchs.append(full_match)
        if len(col_matchs) == 1:
            return col_matchs[0]
        else:
            hudun = [False for i in range(len(col_matchs[0]))]
            match = [[] for i in range(len(col_matchs[0]))]
            for k,cm in enumerate(col_matchs):
                for i,c in enumerate(cm):
                    if c:
                        if input_match and k != 0 and input_match[i] and is_not_all_tables:
                            if i + 2 < len(cm) and ts.tokens[i+1].text == "of" and table_match[i+2] and schema.column_names_original[c[0][0]][0] in table_match[i+2] and col_match_not_in_table(input_match[i],table_match[i+2],schema):
                                input_match[i] = []
                                match[i] = []
                            elif i + 3 < len(cm) and ts.tokens[i+1].text == "all" and ts.tokens[i+2].text == "of" and table_match[i+3] and schema.column_names_original[c[0][0]][0] in table_match[i+3] and col_match_not_in_table(input_match[i],table_match[i+3],schema):
                                input_match[i] = []
                                match[i] = []
                            elif i - 1 >= 0 and table_match[i-1] and schema.column_names_original[c[0][0]][0] in table_match[i-1] and 1 in c[2]  and col_match_not_in_table(input_match[i],table_match[i-1],schema):
                                input_match[i] = []
                                match[i] = []
                            else:
                                continue
                        if input_match and k != 0 and i + 1 < len(table_match) and table_match[i] and input_match[i+1] and schema.column_tokens_table_idx[input_match[i+1][0][0]] in table_match[i]  and is_not_all_tables:
                            continue
                        if input_match and k != 0 and i > 2 and table_match[i] and ts.tokens[i-1].text == "of" and input_match[i-2] and schema.column_tokens_table_idx[input_match[i-2][0][0]] in table_match[i]  and is_not_all_tables:
                            continue
                        if input_match and k != 0 and i > 3 and table_match[i] and ts.tokens[i-1].text == "all" and ts.tokens[i-2].text == "of" and input_match[i-3] and schema.column_tokens_table_idx[input_match[i-3][0][0]] in table_match[i]  and is_not_all_tables:
                            continue
                        if i + 2 < len(cm) and ts.tokens[i+1].text == "of" and table_match[i+2] and schema.column_names_original[c[0][0]][0] in table_match[i+2] and 1 in c[2]:
                            if not hudun[i] and col_match_not_in_table(match[i],table_match[i+2],schema) and (i==0 or col_match_not_in_table(match[i],table_match[i-1],schema)):
                                match[i] = []
                            hudun[i] = True
                        elif i + 3 < len(cm) and ts.tokens[i+1].text == "of" and ts.tokens[i+2].text == "all" and table_match[i+3] and schema.column_names_original[c[0][0]][0] in table_match[i+3] and 1 in c[2]:
                            if not hudun[i] and col_match_not_in_table(match[i],table_match[i+3],schema) and (i==0 or col_match_not_in_table(match[i],table_match[i-1],schema)):   
                                match[i] = []
                            hudun[i] = True
                        elif  i - 1 >= 0 and  table_match[i-1] and schema.column_names_original[c[0][0]][0] in table_match[i-1] and 1 in c[2]:
                            if not hudun[i] and match[i] and col_match_not_in_table(match[i],table_match[i-1],schema) and max(c[1]) >= max(match[i][1]) and (i + 2 >= len(cm) or ts.tokens[i+1].text != "of"):
                                match[i] = []
                            hudun[i] = True
                        elif  i - 2 >= 0 and ts.tokens[i-1].text in ["with","'s"] and  table_match[i-2] and schema.column_names_original[c[0][0]][0] in table_match[i-2] and 1 in c[2]:
                            if not hudun[i] and col_match_not_in_table(match[i],table_match[i-2],schema) and match[i] and max(c[1]) >= max(match[i][1]):
                                match[i] = []
                            hudun[i] = True
                        elif ts.tokens[i].lemma_ == "name" and match[i] and max(c[1]) < max(match[i][1]) and max(match[i][2]) < 1:
                            continue
                        elif ts.tokens[i].lemma_ == "id" and match[i] and max(c[1]) < max(match[i][1]) and max(match[i][2]) < 1:
                            continue
                        elif ts.tokens[i].lemma_ == "name" and match[i] and max(c[1]) > max(match[i][1]) and max(match[i][2]) < 1:
                            if not hudun[i]:
                                match[i] = []
                        elif ts.tokens[i].lemma_ == "id" and match[i] and max(c[1]) > max(match[i][1]) and max(match[i][2]) < 1:
                            if not hudun[i]:
                                match[i] = []
                        elif hudun[i]:
                            continue

                        for d1,d2,d3,d4 in zip(c[0],c[1],c[2],c[3]):
                            if ((input_match and k != 0 and input_match[i]) or table_match[i]) and  d3 == 0:
                                if not special_case_for_pass(d1,schema,ts,i,select_type=select_type):
                                    continue
                            if match[i] and d1 not in match[i][0]:
                                match[i][0].append(d1)
                                match[i][1].append(d2)
                                match[i][2].append(d3)
                                match[i][3].append(d4)
                            elif not match[i]:
                                match[i].append([d1])
                                match[i].append([d2])
                                match[i].append([d3])
                                match[i].append([d4])
            
            if input_match:
                for i, col_in,col_all in zip(range(len(match)),input_match,match):
                    if col_all and not col_in and len(col_all[0]) > 3:
                        for c,nn,kk in zip(reversed(match[i][0]),reversed(match[i][1]),range(len(match[i][0])-1,-1,-1)):
                            if nn == 1:
                                nn = 0
                                for tok in schema.column_tokens[c]:
                                    if " " + tok.lemma_ + " " in " "+ts.lemma_+" ":
                                        nn += 1
                                if nn > 1:
                                    continue
                            else:
                                continue
                            del match[i][0][kk]
                            del match[i][1][kk]
                            del match[i][2][kk]
                            del match[i][3][kk]
                        if match[i] == [[],[],[],[]]:
                            match[i] = []
            if not input_match and len(tables) > 1 and len(match) > 3:
                for i, col_all in zip(range(len(match)),match):
                    remove = True
                    if table_match[i] and col_all and ((i > 2  and ts.tokens[i-1].text == "of" and match[i-2] and schema.column_tokens_table_idx[match[i-2][0][0]] in table_match[i]) or (i > 3  and ts.tokens[i-1].text == "all" and ts.tokens[i-2].text == "of" and match[i-3] and schema.column_tokens_table_idx[match[i-3][0][0]] in table_match[i])):
                        for col in col_all[3]:
                            if (ts.tokens[i-1].text == "of" and schema.column_tokens_table_idx[match[i-2][0][0]] not in table_match[i]) or (ts.tokens[i-2].text == "of" and schema.column_tokens_table_idx[match[i-3][0][0]] not in table_match[i]):
                                remove = False
                                break
                        if remove:
                            match[i] = []
                    elif table_match[i] and col_all and i + 1 < len(table_match) and match[i+1] and schema.column_tokens_table_idx[match[i+1][0][0]] in table_match[i] :
                        for col in col_all[3]:
                            if schema.column_tokens_table_idx[match[i+1][0][0]] not in table_match[i]:
                                remove = False
                                break
                        if remove:
                            match[i] = []
            return match
    def col_or_match(ts,tables,match,schema):
        check_str = " " + ts.text + " "
        for t in tables:
            for i,col in enumerate(schema.tbl_col_tokens_text_str_ori[t]):
                if "*" not in col and " " + col + " " in ts.text:
                    idx_l = check_str.index(" " + col + " ")
                    idx_l = check_str[:idx_l].count(" ")
                    if not match[idx_l]:
                        match[idx_l] = [[],[],[],[]]
                    if i not in match[idx_l][0]:
                        match[idx_l][0].append(schema.tbl_col_idx_back[t][i])
                        match[idx_l][1].append(col.count(" ")+1)
                        match[idx_l][2].append(1)
                        match[idx_l][3].append(schema.tbl_col_idx_back[t][i])
        return match


    match = col_match(tables,ts,schema,table_match, select_type= select_type)
    if all_tables:
        new_tables = [i for i in all_tables if i not in tables]
        if new_tables:
            match = col_match(new_tables,ts,schema,table_match,match,only_exact_match= not select_type, is_not_all_tables=False, select_type=select_type)
        new_tables = [i for i in range(len(schema.table_names_original)) if i not in tables and i not in all_tables]
    else:
        new_tables = [i for i in range(len(schema.table_names_original)) if i not in tables]

    if match.count([]) == len(match) and -1 not in tables and new_tables:
        match = col_match(new_tables,ts,schema,table_match,select_type=select_type)
    elif -1 not in tables and new_tables:
        match = col_match(new_tables,ts,schema,table_match,match,only_exact_match= not select_type,select_type=select_type)
    if match.count([]) == len(match) and tables:
        match = col_or_match(ts,tables,match,schema)
    return match