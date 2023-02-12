import json
import re
from .TokenString import TokenString
from .others_pattern import pattern_reconize,pattern_recomand,one_col_match,DISTINCT_WORDS
from .col_match import of_for_structure_in_col, col_match_main
from .utils import look_for_table_idx, construct_select_data
from .Schema_Token import Schema_Token





def is_no_table_match(table_match):
    """
        Return True, if we can not find table from the question.
    """
    for tm in table_match:
        if tm:
            return False
    return True

def table_match_num(table_match):
    """
        Return the number of tok in question match to table
    """
    count = []
    num = 0
    for tm in table_match:
        if tm:
            for t in tm:
                if t not in count:
                    num += 1
                    count.extend(tm)
                    break
    return num


def use_column_find_table(availble_sent, availble_table, availble_idx, original_sent, original_table, original_idx, schema):
    def check_col_in_the_same_table(agg_cols,schema):
        table_list = []
        for ac in agg_cols:
            if ac and len(ac[1]) == 1:
                table_list.append(schema.column_tokens_table_idx[ac[1][0]])
            else:
                return False,-1
        if table_list:
            last_table_idx = table_list[0]
        else:
            return False,-1
        for t in table_list:
            if t != last_table_idx:
                return False,-1
        return True,last_table_idx

    table_idxs = [-1]
    all_tables = set()
    agg_cols,_1,_2,_3,distinct = type_1( availble_sent, availble_table, availble_idx, original_sent, original_table, original_idx, schema,None,None,table_idxs)
    if agg_cols:
        all_tables = set()
        for ac in agg_cols:
            for c in ac[1]:
                all_tables.add(schema.column_tokens_table_idx[c])
    
    if len(all_tables) == 1:
        return list(all_tables)[0]
    return -1

    for i in range(len(colums)):
        if i>idx_start and colums[i][1] and len(colums[i][1]) == 1:
            return colums[i][1]
    return []



def check_col_correct(column_all):
    for c in column_all:
        if len(c[1]) != 1:
            return False
    return True



def type_1( availble_sent, availble_table, availble_idx, original_sent, original_table, original_idx,
            schema:Schema_Token,sub_q,q_sql,
            table_idxs = [-1]):
    """
    only one table in the availble_table and it will be in the end of the sentence, 
    and we can find every word for it:
    """

    column_all = []
    where_all = []
    group_all = []
    order_all = []
    re_disctinct = False
    for col,a_col,a_col_idx,o_col_idx,col_table,c_idx in zip(original_sent,availble_sent,availble_idx,original_idx,original_table,range(len(original_sent))):
        agg_id, col_id, table_idxs, distinct, where, group, order = one_col_match(col,a_col,a_col_idx,o_col_idx,col_table,schema,table_idxs,availble_sent,c_idx,original_sent,q_sql)
          
        if agg_id != -3:
            column_all.append([agg_id,col_id])
        if where:
            where_all.extend(where)
        if group:
            group_all.append(group)
        if order:
            order_all.append(order)
        if agg_id == 0 and distinct:
            re_disctinct = True
    for i, col in  enumerate(column_all): # for "max and min age".
        if not col[1] and col[0] > 0:
            col[1] = next_col(column_all,i)
    return column_all,where_all,group_all,order_all,re_disctinct
 
def type_2( availble_sent, availble_table, availble_idx, original_sent, original_table, original_idx,
            schema:Schema_Token,sub_q,q_sql,
            table_idx = [-1]):
    column_all,_1,_2,_3,distinct = type_1(availble_sent, availble_table, availble_idx, original_sent, original_table, original_idx, schema, sub_q, q_sql, table_idx)
    if not check_col_correct(column_all):
        for tm,type_ in zip(sub_q.table_match,sub_q.sub_sequence_type):
            if type_ != 0:
                continue
            for ts in tm:
                if ts:
                    table_idx = ts
                    column_all,_1,_2,_3,distinct = type_1(availble_sent, availble_table, availble_idx, original_sent, original_table, original_idx, schema, sub_q, q_sql, table_idx)
                    if check_col_correct(column_all):
                        return column_all,_1,_2,_3,distinct

        for tm,type_ in zip(sub_q.table_match,sub_q.sub_sequence_type):
            if type_ == 0:
                continue
            for ts in tm:
                if ts:
                    table_idx = ts
                    column_all,_1,_2,_3,distinct = type_1(availble_sent, availble_table, availble_idx, original_sent, original_table, original_idx, schema, sub_q, q_sql, table_idx)
                    if check_col_correct(column_all):
                        return column_all,_1,_2,_3,distinct
    return column_all,_1,_2,_3,distinct

def change_table_match(col_idx,old_v,new_v,list_idx,availble_table,original_table,original_idx,sub_q,q_sql):
    for ats in availble_table[col_idx]:
        for at in ats:
            for t in reversed(range(len(at))):
                if at[t] == old_v:
                    if new_v < 0:
                        del at[t]
                    else:
                        at[t] = new_v
    for at in original_table[col_idx]:
        for t in reversed(range(len(at))):
            if at[t] == old_v:
                if new_v < 0:
                    del at[t]
                else:
                    at[t] = new_v
    for idx in original_idx[col_idx]:
        for t in reversed(range(len(q_sql.table_match[idx]))):
            if q_sql.table_match[idx][t] == old_v:
                if new_v < 0:
                    del q_sql.table_match[idx][t]
                else:
                    at[t] = new_v
    for idx,tms in zip(sub_q.original_idx[list_idx],sub_q.table_match[list_idx]):
        if idx in original_idx[col_idx]:
            for t in reversed(range(len(tms))):
                if tms[t] == old_v:
                    if new_v < 0:
                        del tms[t]
                    else:
                        tms[t] = new_v

def pre_analyse_table_match(stemmer, availble_sent, availble_table, availble_idx, original_sent,original_table,original_idx,list_idx,sub_q,q_sql,schema,table_idxs=[-1]):
    column_all = []
    where_all = []
    for col,a_col,a_col_idx,o_col_idx,col_table,c_idx in zip(original_sent,availble_sent,availble_idx,original_idx,original_table,range(len(original_sent))):
        agg_id, col_id, new_table_idx, distinct, where, group, order = one_col_match(col,a_col,a_col_idx,o_col_idx,col_table,schema,table_idxs,availble_sent,c_idx,original_sent,q_sql)
        if col_id and len(col_id) == 1 and schema.column_tokens_table_idx[col_id[0]] != new_table_idx:
            change_table_match(c_idx,new_table_idx,-1,list_idx,availble_table,original_table,original_idx,sub_q,q_sql)



def analyse_table_match(availble_sent,availble_table,availble_idx,original_sent,original_table,original_idx,schema,sentences,sub_q, q_sql):
    """
    remove some wrong table match
    """
    def remove_subq_qsql(sub_q, q_sql, idx):
        q_sql.table_match[idx] = []
        for k,idxs in enumerate(sub_q.original_idx):
            for j,i in enumerate(idxs):
                if i == idx:
                    sub_q.table_match[k][j] = []
                    return sub_q, q_sql
        return sub_q, q_sql

    remove_idx = []
    for at,ai,ast,i in zip(availble_table,availble_idx,availble_sent,range(len(availble_idx))):
        if len(at) == 1 and (len(at[0]) == 1 or (len(at[0]) == 2 and ast[0][0] in DISTINCT_WORDS)) and at[0] and at[0][-1] and schema.column_contain_word(ast[0][-1]) and not schema.column_contain_word("name",at[0][-1]) and sub_q.sub_sequence_type.count(0)==1:   
            availble_table[i][0][-1] = []
            remove_idx.append(availble_idx[i][0][-1])
        if len(at) == 2:
            for t,word,j in zip(at[0],ast[0],range(len(at[0]))): # table of table
                if t and schema.column_contain_word(word):
                    availble_table[i][0][j] = []
                    remove_idx.append(availble_idx[i][0][j])
        if len(at) == 1 and len(at[0]) >= 2 and at[0][-2] and at[0][-1] and at[0][-2] != at[0][-1]:  # table table:
            availble_table[i][0][-1] = []
            remove_idx.append(availble_idx[i][0][-1])
        if len(at) == 1 and len(at[0]) >= 2 and at[0][0] and at[0][1] and at[0][0] != at[0][1]:  # table table:
            availble_table[i][0][1] = []
            remove_idx.append(availble_idx[i][0][1])
    for ot,oi,j in zip(original_table,original_idx,range(len(original_idx))):
        for i,idx in enumerate(oi):
            if idx in remove_idx:
                original_table[j][i] = []
                sub_q, q_sql = remove_subq_qsql(sub_q, q_sql, idx)

    # delete which "table" (actually the "table" is a col)
    if len(sentences) == 1 and len(availble_table) == 1 and len(original_table) == 1 and len(original_table[0])==1 and original_table[0][0]:
        cannot_delete = False
        for t in original_table[0][0]:
            if schema.column_contain_word("name",t):
                cannot_delete = True
        if not cannot_delete:
            agg,col = schema.one_word_to_tables_column_match([-1], original_sent[0][0],cross_table_search=True,use_concept_match=False)
            if len(col) > 0:
                availble_table = [[[[]]]]
                original_table = [[[]]]
                sub_q, q_sql = remove_subq_qsql(sub_q, q_sql, original_idx[0][0])
                
    return availble_table,original_table,sub_q, q_sql 


def reconize_select_type(stemmer, availble_sent, availble_table, availble_idx, original_sent,original_table,original_idx,list_idx,sub_q,q_sql,schema):
    
    def strict_type_1(original_table):
        for col in original_table:
            for ts in col:
                if len(ts) > 1:
                    return False
                for t in ts:
                    if t != original_table[-1][-1][-1]:
                        return False
        return True
    
    def close_to_type_1(original_idx,q_sql):
        if is_no_table_match(sub_q.table_match[list_idx]):
            idx = original_idx[-1][-1]
            if idx + 2 < len(q_sql.table_match):
                if len(q_sql.table_match[idx + 2]) == 1:
                    return (True,q_sql.table_match[idx + 2])
        return (False,[-1])

    def only_one_table(tm_num,original_table,availble_table,availble_idx,q_toks,list_idx,sub_q):
        if list_idx > 0 and sub_q.sub_sequence_list[0].startswith("for each"): 
            for tm in sub_q.table_match[0]:
                if tm:
                    return True,tm
        if tm_num == 1:
            if original_table[-1][-1]:
                return True,original_table[-1][-1]
            elif original_table[0][0]:
                return True,original_table[0][0]
        elif tm_num == 2:
            table_idx = [-1]
            table_in_col_idx = -1
            for i, col,a_col in zip(range(len(original_table)),original_table,availble_idx):
                for c in col:
                    if c != []:
                        if len(c) >= 1:
                            if table_idx == [-1]:
                                table_idx = c
                                table_in_col_idx = i
                            elif table_in_col_idx == i and of_for_structure_in_col(a_col,q_toks):
                                table_idx = c
            return True,table_idx
        return False, [-1]

    def one_table_per_col(original_table):
        t_in_c = [False]*len(original_table)
        for i, col in enumerate(original_table):
            for c in col:
                if c != []:
                    if len(c) == 1:
                        t_in_c[i] = True
                    elif len(c) > 1:
                        return False
        if False in t_in_c:
            return False
        return True

    tm_num = table_match_num(sub_q.table_match[list_idx])
    oot, table_idxs = only_one_table(tm_num,original_table,availble_table,availble_idx,q_sql.question_tokens,list_idx,sub_q)
    if table_idxs[0] >= 0 or one_table_per_col(original_table):
        if table_idxs[0] < 0 and len(original_table)==1:
            for t in original_table[0]:
                if t:
                    table_idxs = t
        return type_1,table_idxs
    if is_no_table_match(q_sql.table_match): # Special type 1. No table in table_match but it is high possible the table is in the end of select which is the same as type 1
        if len(availble_sent[-1]) == 2:
            check = True
            for idx in availble_idx[-1][-1]:
                if sub_q.question_tokens[list_idx][idx-sub_q.offset[list_idx]].text in schema.table_col_text[-1] or sub_q.question_tokens[list_idx][idx-sub_q.offset[list_idx]].lemma_ in schema.table_col_text[-1]  or sub_q.question_tokens[list_idx][idx-sub_q.offset[list_idx]].lemma_ in schema.table_col_lemma[-1]:
                    check = False
            if check:
                # Try to find the table through column name
                # table_idx is not a list.
                table_idx = use_column_find_table(availble_sent, availble_table, availble_idx, original_sent, original_table, original_idx, schema)
                if table_idx >= 0:
                    availble_table[-1][-1]=[[table_idx] for i in range(len(availble_sent[-1][-1]))]
                    for i in range(len(availble_sent[-1][-1])):
                        original_table[-1][-1-i] = [table_idx]

                    for i in availble_idx[-1][-1]:
                        q_sql.table_match[i] = [table_idx]
                        for k,idx in enumerate(sub_q.original_idx[list_idx]):
                            if idx == i:
                                sub_q.table_match[list_idx][k] = [table_idx]
                                break
                    return type_1,[table_idx]
    if close_to_type_1(original_idx,q_sql)[0]: 
        _,table_idxs = close_to_type_1(original_idx,q_sql)
        if strict_type_1(original_table):
            return type_1,table_idxs
    if tm_num <= 1: # two select sentence with one table type
        table_idxs = [-1]
        if tm_num:
            for tm in sub_q.table_match[list_idx]:
                if tm:
                    table_idxs = tm
                    break
        return type_2,table_idxs
    return type_2,[-1]


def combine_sgwo(re_value,select_key_word):
    key = None
    select,where,group,order,distinct = (None,None,None,None,False)
    for i,c in enumerate(re_value):
        if not select and c[0]:
            select = c[0]
            key = select_key_word[i]
        elif c[0]:
            for s,skw in zip(c[0],select_key_word[i]):
                all_in_previous = False
                if len(key) == 1: 
                    all_in_previous = True
                    for w in skw:
                        if w not in key[0] or w == "each":
                            all_in_previous = False
                            break
                if all_in_previous:
                    continue
                if s not in select:
                    select.append(s)
        if c[1]:
            where = c[1]
        if c[2]:
            group = c[2]
        if c[3]:
            order = c[3]
        if c[4]:
            distinct = True

    return select,where,group,order,distinct


def others_analyze(sentences, table_matchs, list_idxs, schema, sub_q, q_sql, select_type=False, in_db_match=None, final_select=False, no_pattern=False):
    pattern_type = []
    others = []
    pattern_unknow = []
    others_token_pattern = []
    if in_db_match:
        in_db_match2 = in_db_match
    else:
        in_db_match2 = [None] * len(sentences)
    for sentence, table_match, list_idx, db_m in zip(sentences,table_matchs,list_idxs, in_db_match2):
        tables = look_for_table_idx(sub_q, list_idx, 1, schema)
        if final_select and select_type and sub_q.sub_sequence_type[list_idx] == 1:
            for cols in sub_q.col_match[list_idx]:
                if cols:
                    for col in cols[0]:
                        if schema.column_names_original[col][0] not in tables:
                            tables.append(schema.column_names_original[col][0])

        ts = TokenString(None,sub_q.question_tokens[list_idx])
        full_match = col_match_main(tables,ts,schema,table_match,select_type)
        sub_q.col_match[list_idx] = full_match
        if no_pattern:
            continue
        pr,db_match = pattern_reconize(ts,table_match,full_match,sub_q.sequence_entt[list_idx],schema,tables,pattern_toks=[[['START', 'SEARCH', 'DATABASE']]],pattern_fun=[None],in_db_match=db_m)
        others.append(pr[0:4])

        if pr == (None,None,None,None,None,None):
            pattern_unknow.append([pattern_recomand(ts,table_match,full_match,sub_q.sequence_entt[list_idx],db_match,schema,tables)[0:2],None])
        else:
            pattern_unknow.append([(pr[4],pr[5]),(pr[0],pr[1],pr[2],pr[3])])
        others_token_pattern.append(pattern_unknow[-1][0][1])
        sub_q.pattern_tok[list_idx] = pattern_unknow[-1][0][1]
        if len(others_token_pattern[-1]) != len(table_match):
            print("Pattern Error!!!!!!!!!!!!!!!!!")

    return others,pattern_type,others_token_pattern


def select_analyze(stemmer, sentences, table_matchs, list_idxs, schema:Schema_Token, sub_q, q_sql):
    def shall_we_skip_first_select(sentences,original_table,start_idx,list_idxs,sub_sentence_idx):
        skip_table_set = set()
        skip_first_select = True
        for i in range(start_idx+1,len(list_idxs)):
            if "also " in sentences[i] and sub_sentence_idx[list_idxs[i]]:
                skip_first_select = False
        for ot in original_table[0]:
            if not ot:
                pass
            else:
                for t in ot:
                    skip_table_set.add(t)
        return skip_first_select,skip_table_set

    select,where,group,order = (None,None,None,None)
    sentence_num = len(sentences)
    real_sentence_num = (q_sql.question + " ").count(" . ")+(q_sql.question + " ").count(" ? ")
    re_value = []
    key_word = []
    skip_first_select = False
    skip_table_set = set()
        
    for i, sentence,table_match,list_idx in zip(range(sentence_num), sentences, table_matchs, list_idxs):
        
        availble_sent,availble_table,availble_idx,original_sent,original_table,original_idx = construct_select_data(sentence, table_match, sub_q.original_idx[list_idx],schema)
        if  len(original_sent) == 1 and not original_sent[0]:
            continue
        elif availble_sent and ("each" in availble_sent[0][0] or ( len(availble_sent[0][0]) == 1 and " by each " in sentence) ):
            continue
        if sub_q.sub_sentence_idx[list_idx] == 0 and real_sentence_num == 2 and len(original_table) == 1:
            skip_first_select,skip_table_set = shall_we_skip_first_select(sentences,original_table,i,list_idxs,sub_q.sub_sentence_idx)
            if skip_first_select:
                continue
        elif real_sentence_num == 2 and sentence_num == 3 and i < 2 and " each " in sentence:
            continue

        fun,table_idxs = reconize_select_type(stemmer, availble_sent,availble_table,availble_idx, original_sent, original_table, original_idx, list_idx, sub_q, q_sql, schema)
        if not fun:
            pre_analyse_table_match(stemmer, availble_sent,availble_table,availble_idx, original_sent, original_table, original_idx, list_idx, sub_q, q_sql, schema,table_idxs)
            fun,table_idxs = reconize_select_type(stemmer, availble_sent,availble_table,availble_idx, original_sent, original_table, original_idx, list_idx, sub_q, q_sql, schema)
        if fun:
            if skip_first_select and len(skip_table_set) > 0:
                table_idxs = list(skip_table_set)
            re_value.append(fun(availble_sent, availble_table, availble_idx, original_sent, original_table, original_idx, schema, sub_q, q_sql, table_idxs))
            key_word.append(original_sent)
        print(sentence)
        print(availble_sent)

    if len(re_value) == 1:
        select,where,group,order,distinct = re_value[0]
    else:
        select,where,group,order,distinct = combine_sgwo(re_value,key_word)
    
    if sentences and (  (" each " in sentences[0] or (len(sentences) > 1 and " each " in sentences[1]) or (len(sentences) > 2 and " each " in sentences[2]))  or ((" for the " in sentences[0] or (len(sentences) > 1 and " for the " in sentences[1]) or (len(sentences) > 2 and " for the " in sentences[2])) and len(select) == 1 and select[0][0]) ) :
        if not (" each " in sentences[0] or (len(sentences) > 1 and " each " in sentences[1]) or (len(sentences) > 2 and " each " in sentences[2])):
            for i, sentence in enumerate(sentences):
                sentences[i] = sentences[i].replace(" for the "," for each ")
        if not select or len(select) == 1 or (len(select) == 2 and (select[0][0] and select[1][0])):
            for i, sentence,table_match,list_idx in zip(range(sentence_num), sentences, table_matchs, list_idxs):
                if  " each " not in sentence:
                    continue
                if select and not select[0][0] and real_sentence_num == 2 and sentence_num == 3 and i < 2 and " each " in sentence:
                    continue
                availble_sent,availble_table,availble_idx,original_sent,original_table,original_idx = construct_select_data(sentence, table_match, sub_q.original_idx[list_idx],schema)
                if  (len(original_sent) == 1 and not original_sent[0]) or "each" not in original_sent[0]:
                    continue

                availble_table,original_table, sub_q, q_sql  = analyse_table_match(availble_sent,availble_table,availble_idx,original_sent,original_table,original_idx,schema,sentences, sub_q, q_sql)
                
                if i == 0 and real_sentence_num == 2 and sentence_num == 2 and len(original_table) == 1:
                    skip_first_select,skip_table_set = shall_we_skip_first_select(sentences,original_table,i,list_idxs,sub_q.sub_sentence_idx)
                    if skip_first_select:
                        continue
                
                if not select or select[0][0] or (len(select) == 2 and (select[0][0] and select[1][0])):
                    beak_each = False
                    for k,asts in enumerate(availble_sent):
                        for x,ast in enumerate(asts):
                            if "each" in ast:
                                availble_sent = [availble_sent[k][x:]]
                                availble_table = [availble_table[k][x:]]
                                availble_idx = [availble_idx[k][x:]]
                                beak_each = True
                                break
                        if beak_each:
                            break
                    if beak_each:
                        len_each = 0
                        for x,ast in enumerate(availble_sent[0]):
                            len_each += len(ast)
                        start_idx_or = len(original_sent[k])-len_each
                        original_sent = [original_sent[k][start_idx_or:]]
                        original_table = [original_table[k][start_idx_or:]]
                        original_idx = [original_idx[k][start_idx_or:]]
                    
                    fun,table_idxs = reconize_select_type(stemmer, availble_sent,availble_table,availble_idx, original_sent, original_table, original_idx, list_idx, sub_q, q_sql, schema)
                    if not fun:
                        pre_analyse_table_match(stemmer, availble_sent,availble_table,availble_idx, original_sent, original_table, original_idx, list_idx, sub_q, q_sql, schema,table_idxs)
                        fun,table_idxs = reconize_select_type(stemmer, availble_sent,availble_table,availble_idx, original_sent, original_table, original_idx, list_idx, sub_q, q_sql, schema)
                    if fun:
                        if skip_first_select and len(skip_table_set) > 0:
                            table_idxs = list(skip_table_set)
                        re_value.append(fun(availble_sent, availble_table, availble_idx, original_sent, original_table, original_idx, schema, sub_q, q_sql, table_idxs))
                        key_word.append(original_sent)
                    print(sentence)
                    print(availble_sent)
                    if len(re_value) == 1:
                        return re_value[0]
                    else:
                        return combine_sgwo(re_value,key_word)
                else:
                    index_each = original_sent[0].index("each")
                    for k,ts in enumerate(original_table[0]):
                        if len(ts)==1 and k > index_each:
                            no_add = False
                            for c in select[0][1]:
                                if schema.column_names_original[c][0] == ts[0]:
                                    no_add = True
                            if not no_add:
                                pk = schema.primary_keys(ts[0])
                                if pk > 0:
                                    return select+[[0,[pk]]],where,group,order,distinct
        
    return select,where,group,order,distinct


