import json
import argparse
import pickle
import copy,os
from typing import Callable, Dict, List, Set
import sqlite3
import re
from natsql2sql.preprocess.TokenString import get_spacy_tokenizer,SToken,TokenString
from natsql2sql.preprocess.sq import SubQuestion,QuestionSQL
from natsql2sql.preprocess.table_match import return_column_match
from natsql2sql.preprocess.others_pattern import get_AWD_column
from natsql2sql.preprocess.match import AGG_WORDS,AGG_OPS,STOP_WORDS, S_ADJ_WORD_DIRECTION, ABSOLUTELY_GRSM_DICT, ALL_JJS
from natsql2sql.preprocess.col_match import of_for_structure_in_col,col_match_main
from natsql2sql.preprocess.utils import look_for_table_idx, construct_select_data,is_there_sgrsm_and_gr_or_sm,sjjs_table,get_all_table_from_sq,get_all_col_from_sq,str_is_date
from natsql2sql.preprocess.stemmer import MyStemmer
from natsql2sql.preprocess.Schema_Token import Schema_Token
from natsql2sql.preprocess.pattern_analyze import others_analyze,select_analyze



def construct_hyper_param():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_file', default="dev-or.json", type=str)
    parser.add_argument('--out_file', default='dev-or-word.json', type=str)
    parser.add_argument("--table_file", default='NatSQLv1_6/tables.json', type=str, help="table json file")
    parser.add_argument("--keep_or", action='store_true', default=False)
    args = parser.parse_args()
    return args



def before_or_after(tps,idx,target):
    if idx > 0 and tps[idx-1] == target:
        return True
    elif idx + 1 < len(tps) and tps[idx+1] == target:
        return True
    return False

def not_in_all_q(question,str_):
    for q in question:
        if str_ in q:
            return False
    return True

def jjs_not_order(real_tp_str,tp_str2):
    # two col
    if "AGG and JJS" not in tp_str2 and "AGG and the JJS" not in tp_str2 and "JJS and AGG" not in tp_str2 and "JJS and the JJS" not in tp_str2 and "JJS and JJS" not in tp_str2 and "JJS SC and JJS S" not in tp_str2 and "JJS STC and JJS S" not in tp_str2  and "AGG SC and JJS S" not in tp_str2:
        pass
    else:
        return False
    real_tp_str = real_tp_str.replace("GR_JJS","JJS")
    real_tp_str = real_tp_str.replace("SM_JJS","JJS")
    if "AGG SC JJS SC JJS S" in real_tp_str or "AGG SC AGG SC JJS S" in real_tp_str  or "AGG SC JJS SC AGG S" in real_tp_str or "JJS SC AGG SC AGG S" in real_tp_str or "JJS SC AGG SC JJS S" in real_tp_str or "JJS SC JJS SC AGG S" in real_tp_str or "JJS SC JJS SC JJS S" in real_tp_str:
        return False
    return True

def others_where_order_analyse(token_pattern,question,select_q,select):
    where = 0
    order = 0
    limit = 0
    desc  = 0
    return_result =[ [0,0,0,0] for i in token_pattern ] # order limit desc(1)/asc 
    for i,tps in enumerate(token_pattern):
        tp_str = " ".join(tps)
        real_tp_str = " ".join([ t for t in tps if t not in ["#","?"] ])
        idx_jjr = -1
        if ('IN GR_JJS COL NUM' in real_tp_str or "from GR_JJS GR_JJS NUM" in real_tp_str or "from GR_JJS NUM" in real_tp_str) and ("first" in question[i] or "one" in question[i]):
            return_result[i][0] = 1
            return_result[i][2] = 1
        elif ('IN SM_JJS COL NUM' in real_tp_str or "from SM_JJS SM_JJS NUM" in real_tp_str or "from SM_JJS NUM" in real_tp_str) and ("first" in question[i] or "one" in question[i]):
            return_result[i][0] = 1
        elif "from " in real_tp_str and (' SM_JJS GR_JJS' in real_tp_str or ' SM_GRSM GR_GRSM' in real_tp_str or ' SM_JJS GR_GRSM' in real_tp_str or ' SM_GRSM GR_JJS' in real_tp_str):
            return_result[i][0] = 1
        elif "from " in real_tp_str and (' GR_JJS SM_JJS' in real_tp_str or ' GR_GRSM SM_GRSM' in real_tp_str or ' GR_JJS SM_GRSM' in real_tp_str or ' GR_GRSM SM_JJS' in real_tp_str):
            return_result[i][0] = 1
            return_result[i][2] = 1
        else:
            idx_jjr = 0
            q_tok = ""
            for (j,p),qtok in zip(enumerate(tps),question[i].split(" ")):
                if p in ["SM_SJJS","SM_JJS","GR_SJJS","GR_JJS"]:
                    q_tok = qtok
                    if q_tok in ALL_JJS.keys():
                        q_tok = ALL_JJS[q_tok]
            if 'GR_JJS' in tps:
                idx_jjr = tps.index('GR_JJS')
            elif 'SM_JJS' in tps:
                idx_jjr = tps.index('SM_JJS')

            if tps[0] == 'order' or 'order by' in question[i] or 'ordered by' in question[i] or "in order" in question[i] or "in the order of " in question[i] or "order the result by " in question[i]  or "ordered the result by " in question[i]  or "order the results by " in question[i]  or "ordered the results by " in question[i]:
                return_result[i][0] = 1
            if 'descend' in tps or 'descending' in tps :
                return_result[i][0] = 1
                return_result[i][2] = 1
            elif 'ascend' in tps or 'alphabetical' in tps or 'ascending' in tps or "lexicographic" in tps:
                return_result[i][0] = 1
            elif "NOT" not in tps and idx_jjr != -1 and 'GR_JJS' in tps and (("GRSM" not in tp_str and " at most " not in question[i]) or before_or_after(tps,idx_jjr,"NUM")) and not (real_tp_str.count(" ") <= 2 and not not_in_all_q(select_q," each ")) and ("COL" in real_tp_str or "TABLE" in real_tp_str or len(select)>1 or question[i].split(" ")[0] in ["that","who","which","when"] or q_tok in S_ADJ_WORD_DIRECTION.keys()):
                return_result[i][0] = 1
                return_result[i][1] = 1
                return_result[i][2] = 1
            elif "NOT" not in tps and idx_jjr != -1 and 'SM_JJS' in tps and (("GRSM" not in tp_str and " at least " not in question[i]) or before_or_after(tps,idx_jjr,"NUM"))and not (real_tp_str.count(" ") <= 2 and not not_in_all_q(select_q," each ")) and ("COL" in real_tp_str or "TABLE" in real_tp_str or len(select)>1 or question[i].split(" ")[0] in ["that","who","which","when"] or q_tok in S_ADJ_WORD_DIRECTION.keys()):
                return_result[i][0] = 1
                return_result[i][1] = 1
            elif 'top' in tps and "NUM" in tps :
                return_result[i][0] = 1
                return_result[i][1] = 1
                return_result[i][2] = 1
            elif ("GR_JJS GR_JJS" in real_tp_str and (" most frequent" in question[i] or " most common" in question[i] or " most popular" in question[i])) or ("GR_JJS frequent" in real_tp_str) or ("GR_JJS common" in real_tp_str) or ("GR_JJS popular" in real_tp_str):
                return_result[i][0] = 1
                return_result[i][1] = 1
                return_result[i][2] = 1
            elif ("SM_JJS SM_JJS" in real_tp_str and (" least frequent" in question[i] or " least common" in question[i] or " least popular" in question[i])) or ("SM_JJS frequent" in real_tp_str) or ("SM_JJS common" in real_tp_str) or ("SM_JJS popular" in real_tp_str):
                return_result[i][0] = 1
                return_result[i][1] = 1
            elif "SM_JJS AGG " in real_tp_str and " number of " in question[i]:
                return_result[i][0] = 1
                return_result[i][1] = 1
            elif "GR_JJS AGG " in real_tp_str and " number of " in question[i]:
                return_result[i][0] = 1
                return_result[i][1] = 1
                return_result[i][2] = 1
            elif 'JJS' in tps:
                pass
            else:
                if "NOT" not in tps and (" maximum " in question[i] or " minimum " in question[i]) and "than" not in tps and "GRSM" not in tp_str and "AGG" in tps:
                    qtoks = question[i].split(" ")
                    idx_m = qtoks.index("maximum") if " maximum " in question[i] else qtoks.index("minimum")
                    if tps[idx_m] == "AGG":
                        return_result[i][0] = 1
                        return_result[i][1] = 1
                        return_result[i][2] = 1 if " maximum " in question[i] else 0


        if return_result[i][0] == 0 and return_result[i][3] == 0:
            if "order" in tps or 'sort' in tps:
                return_result[i][0] = 1
            elif ' majority of ' in question[i]:
                return_result[i][0] = 1
                return_result[i][1] = 1
                return_result[i][2] = 1
            elif ' most ' in question[i] and "GRSM" not in tp_str and 'at most ' not in question[i]:
                return_result[i][0] = 1
                return_result[i][1] = 1
                return_result[i][2] = 1
            elif ' least ' in question[i] and "GRSM" not in tp_str and 'at least ' not in question[i]:
                return_result[i][0] = 1
                return_result[i][1] = 1

        if return_result[i][1] == 1 and "JJS" in tp_str:
            if  "starting with" not in question[i] and "starting from" not in question[i] and "start with" not in question[i] and "start from" not in question[i] and "starts with" not in question[i] and "starts from" not in question[i] and "started with" not in question[i] and "started from" not in question[i]:
                pass
            else:
                return_result[i][1] = 0

        if return_result[i][0] == 1 and return_result[i][1] == 0:
            if " reverse" in question[i]:
                return_result[i][2] = 1 if return_result[i][2] == 0 else 0

        if return_result[i][0] and "from GR_GRSM" in tp_str:
            return_result[i][2] = 1
        if "start from" not in tp_str and return_result[i][0] and "NUM" in tp_str and 'IN GR_JJS COL NUM' not in real_tp_str and 'IN SM_JJS COL NUM' not in real_tp_str and "from GR_JJS GR_JJS NUM" not in real_tp_str and "from GR_JJS NUM" not in real_tp_str and "from SM_JJS SM_JJS NUM" not in real_tp_str and "from SM_JJS NUM" not in real_tp_str:
            return_result[i][1] = 1
    return return_result

def select_where_order_analyse(token_pattern,question,select):
    where = 0
    order = 0
    limit = 0
    desc  = 0
    return_result =[ [order,limit,desc,where] ] * len(token_pattern)  # order limit desc(1)/asc 
    for i,tps in enumerate(token_pattern):
        tp_str = " ".join(tps)
        tp_str2 = ""
        for tp, q in zip(tps,question[i].split(" ")):
            if tp in ["#","?"]:
                tp_str2 += (q+" ")
            else:
                tp_str2 += (tp+" ")
        tp_str2 = tp_str2[:-1]
        tp_str2 = tp_str2.replace(" GR_JJS "," JJS ")
        tp_str2 = tp_str2.replace(" SM_JJS "," JJS ")
        real_tp_str = " ".join([ t for t in tps if t not in ["#","?"] ])

        if ('IN GR_JJS COL NUM' in real_tp_str or "from GR_JJS GR_JJS NUM" in real_tp_str or "from GR_JJS NUM" in real_tp_str) and ("first" in question[i] or "one" in question[i]):
            return_result[i][0] = 1
            return_result[i][2] = 1
        elif ('IN SM_JJS COL NUM' in real_tp_str or "from SM_JJS SM_JJS NUM" in real_tp_str or "from SM_JJS NUM" in real_tp_str) and ("first" in question[i] or "one" in question[i]):
            return_result[i][0] = 1
        elif "from " in real_tp_str and (' SM_JJS GR_JJS' in real_tp_str or ' SM_GRSM GR_GRSM' in real_tp_str or ' SM_JJS GR_GRSM' in real_tp_str or ' SM_GRSM GR_JJS' in real_tp_str):
            return_result[i][0] = 1
        elif "from " in real_tp_str and (' GR_JJS SM_JJS' in real_tp_str or ' GR_GRSM SM_GRSM' in real_tp_str or ' GR_JJS SM_GRSM' in real_tp_str or ' GR_GRSM SM_JJS' in real_tp_str):
            return_result[i][0] = 1
            return_result[i][2] = 1
        else:
            jjs_num = 0
            col_table_after_jjs = 0
            for j,p in enumerate(tps):
                if j > 0 and tps[j-1] == p:
                    continue
                elif p in ["AGG","SM_SJJS","SM_JJS","GR_SJJS","GR_JJS"]:
                    jjs_num += 1
                if jjs_num and  col_table_after_jjs and p in ["ST","STC","SC","COL","TABLE","TABLE-COL"]:
                    col_table_after_jjs = 2
                elif not jjs_num and not col_table_after_jjs and p in ["ST","STC","SC","COL","TABLE","TABLE-COL"]:
                    col_table_after_jjs = 1

            if tps[0] == 'order' or 'order by' in question[i] or 'ordered by' in question[i] or "in order" in question[i] or "in the order of " in question[i] or "order the result by " in question[i]  or "ordered the result by " in question[i]  or "order the results by " in question[i]  or "ordered the results by " in question[i]:
                return_result[i][0] = 1
            if 'sort' in tps:
                return_result[i][0] = 1
            if 'descend' in tps or 'descending' in tps:
                return_result[i][0] = 1
                return_result[i][2] = 1
            elif 'ascend' in tps or 'alphabetical' in tps or 'ascending' in tps:
                return_result[i][0] = 1
            elif 'GR_JJS' in tps and "GRSM" not in tp_str and " at most " not in question[i] and not_in_all_q(question," each ") and not_in_all_q(question,"for every") and not_in_all_q(question,"For every") and jjs_not_order(real_tp_str,tp_str2) and (len(select)!=1 or col_table_after_jjs==2 or " most common" in question[i]) and jjs_num==1:
                return_result[i][0] = 1
                return_result[i][1] = 1
                return_result[i][2] = 1
            elif 'SM_JJS' in tps and "GRSM" not in tp_str and " at least " not in question[i] and not_in_all_q(question," each ") and not_in_all_q(question,"for every") and not_in_all_q(question,"For every") and jjs_not_order(real_tp_str,tp_str2) and (len(select)!=1 or col_table_after_jjs==2 or " least common" in question[i]) and jjs_num == 1:
                return_result[i][0] = 1
                return_result[i][1] = 1
            elif ('top' in tps and "NUM" in tps) or " top SC by " in tp_str2 or " top STC by " in tp_str2:
                return_result[i][0] = 1
                return_result[i][1] = 1
                return_result[i][2] = 1
            elif ' most common ' in question[i]:
                return_result[i][0] = 1
                return_result[i][1] = 1
                return_result[i][2] = 1
            elif 'NUM GR_JJS' in tp_str or 'NUM GR_SJJS' in tp_str or 'GR_JJS NUM' in tp_str or 'GR_SJJS NUM' in tp_str :
                return_result[i][0] = 1
                return_result[i][1] = 1
                return_result[i][2] = 1
            elif 'NUM SM_JJS' in tp_str or 'NUM SM_SJJS' in tp_str or 'SM_JJS NUM' in tp_str or 'SM_SJJS NUM' in tp_str :
                return_result[i][0] = 1
                return_result[i][1] = 1
            elif (("GR_JJS GR_JJS " in real_tp_str or "most GR_JJS" in real_tp_str) and (" most frequent" in question[i] or " most common" in question[i] or " most popular" in question[i])) or ("GR_JJS frequent" in real_tp_str) or ("GR_JJS common" in real_tp_str) or ("GR_JJS popular" in real_tp_str):
                return_result[i][0] = 1
                return_result[i][1] = 1
                return_result[i][2] = 1
            elif (("SM_JJS SM_JJS " in real_tp_str or "least SM_JJS " in real_tp_str) and (" least frequent" in question[i] or " least common" in question[i] or " least popular" in question[i])) or ("SM_JJS frequent" in real_tp_str) or ("SM_JJS common" in real_tp_str) or ("SM_JJS popular" in real_tp_str):
                return_result[i][0] = 1
                return_result[i][1] = 1
            elif " most " in question[i] and jjs_num == 1 and "GR_JJS JJ S" in real_tp_str:
                return_result[i][0] = 1
                return_result[i][1] = 1
                return_result[i][2] = 1
            elif " least " in question[i] and jjs_num == 1 and "SM_JJS JJ S" in real_tp_str:
                return_result[i][0] = 1
                return_result[i][1] = 1
            elif "SM_JJS AGG " in real_tp_str and " number of " in question[i]:
                return_result[i][0] = 1
                return_result[i][1] = 1
            elif "GR_JJS AGG " in real_tp_str and " number of " in question[i]:
                return_result[i][0] = 1
                return_result[i][1] = 1
                return_result[i][2] = 1
            elif 'NUM S' in real_tp_str and "name" not in question[i] and "grade" not in question[i] and real_tp_str.count("NUM")==1:
                real_tp_str = []
                for t,q in zip(tps,question[i].split(" ")):
                    if t == "NUM":
                        real_tp_str.append(q)
                    elif t not in ["#","?"]:
                        real_tp_str.append(t)
                if "first" in real_tp_str:
                    return_result[i][0] = 1
                    return_result[i][1] = 1
            
        if return_result[i][0] == 1 and return_result[i][1] == 0:
            if " reverse" in question[i]:
                return_result[i][2] = 1 if return_result[i][2] == 0 else 0

    return return_result



def reset_frist_last_col_sq(sq,idx_first,list_idx,i,real_tp_list,schema,next_allow_p_tok,in_col_type,new_word,new_lemma):
    target_idx = -1
    yes = False
    if idx_first + 1 < len(real_tp_list) and real_tp_list[idx_first + 1][0] in next_allow_p_tok:
        target_idx = idx_first + 1
    elif idx_first + 2 < len(real_tp_list) and real_tp_list[idx_first + 1][0] in ["NUM","*"] and real_tp_list[idx_first + 2][0] in next_allow_p_tok:
        target_idx = idx_first + 2
    if target_idx >= 0:
        for col in sq.col_match[list_idx[i]][real_tp_list[target_idx][1]][0]:
            if schema.column_types[col] == in_col_type:
                yes = True
        if yes:
            for k in reversed(range(len(sq.col_match[list_idx[i]][real_tp_list[target_idx][1]][0]))):
                col = sq.col_match[list_idx[i]][real_tp_list[target_idx][1]][0][k]
                if schema.column_types[col] in ["text","number"] and schema.column_types[col] != in_col_type:
                    del sq.col_match[list_idx[i]][real_tp_list[target_idx][1]][0][k]
                    del sq.col_match[list_idx[i]][real_tp_list[target_idx][1]][1][k]
                    del sq.col_match[list_idx[i]][real_tp_list[target_idx][1]][2][k]
                    del sq.col_match[list_idx[i]][real_tp_list[target_idx][1]][3][k]
            q_old = " ".join([tt.text for tt in sq.question_tokens[list_idx[i]]])
            sq.question_tokens[list_idx[i]][real_tp_list[idx_first][1]] = SToken(text=new_word,lemma=new_lemma)
            q_new = " ".join([tt.text for tt in sq.question_tokens[list_idx[i]]])
            print(q_old)
            print(q_new)
            return sq,yes
    return sq,yes

def reset_frist_last_tb_sq(sq,idx_first,list_idx,i,real_tp_list,schema,next_allow_p_tok,new_word,new_lemma):
    target_idx = -1
    if idx_first + 1 < len(real_tp_list) and real_tp_list[idx_first + 1][0] in next_allow_p_tok:
        target_idx = idx_first + 1
    elif idx_first + 2 < len(real_tp_list) and real_tp_list[idx_first + 1][0] in ["NUM","*"] and real_tp_list[idx_first + 2][0] in next_allow_p_tok:
        target_idx = idx_first + 2
    yes = False
    if target_idx >= 0:
        for tbl in sq.table_match[list_idx[i]][real_tp_list[target_idx][1]]:
            for col in schema.tbl_col_idx_back[tbl]:
                if schema.column_types[col] == "time":
                    yes = True
                    break
        if yes:
            q_old = " ".join([tt.text for tt in sq.question_tokens[list_idx[i]]])
            sq.question_tokens[list_idx[i]][real_tp_list[idx_first][1]] = SToken(text=new_word,lemma=new_lemma)
            q_new = " ".join([tt.text for tt in sq.question_tokens[list_idx[i]]])
            print(q_old)
            print(q_new)
            return sq,yes
    return sq,yes





def question_modify(token_pattern,question,list_idx,sq,schema):
    for i,tps in enumerate(token_pattern):
        tp_str = " ".join(tps)
        real_tp_str = " ".join([ t for t in tps if t not in ["#","?"] ])
        real_tp_list = [ (t,j) for j,t in enumerate(tps) if t not in ["#","?"] ]

        idx_jjr = -1
        if ('IN GR_JJS COL NUM' in real_tp_str or "from GR_JJS GR_JJS NUM" in real_tp_str or "from GR_JJS NUM" in real_tp_str  or 'IN GR_JJS GR_JJS NUM' in real_tp_str  or 'IN GR_JJS NUM' in real_tp_str) and ("first" in question[i] or "one" in question[i]):
            if 'IN GR_JJS COL NUM' in real_tp_str:
                idx = real_tp_str.index('IN GR_JJS COL NUM')
                left_str = real_tp_str[:idx]
                idx = left_str.count(" ")
                sq.question_tokens[list_idx[i]][real_tp_list[idx+1][1]] = SToken(text="descending",lemma="descend")
                sq.question_tokens[list_idx[i]][real_tp_list[idx+3][1]] = SToken(text="order",lemma="order") 
                sq.col_match[list_idx[i]][real_tp_list[idx+1][1]] = []
                sq.col_match[list_idx[i]][real_tp_list[idx+3][1]] = []
            elif 'from GR_JJS GR_JJS NUM' in real_tp_str or 'IN GR_JJS GR_JJS NUM' in real_tp_str:
                idx = real_tp_str.index('from GR_JJS GR_JJS NUM')  if 'from GR_JJS GR_JJS NUM' in real_tp_str else real_tp_str.index('IN GR_JJS GR_JJS NUM')
                left_str = real_tp_str[:idx]
                idx = left_str.count(" ")
                sq.question_tokens[list_idx[i]][real_tp_list[idx+1][1]] = SToken(text="descending",lemma="descend")
                sq.question_tokens[list_idx[i]][real_tp_list[idx+2][1]] = SToken(text="to",lemma="to")
                sq.question_tokens[list_idx[i]][real_tp_list[idx+3][1]] = SToken(text="end",lemma="end")
                sq.col_match[list_idx[i]][real_tp_list[idx+1][1]] = []
                sq.col_match[list_idx[i]][real_tp_list[idx+2][1]] = []
                sq.col_match[list_idx[i]][real_tp_list[idx+3][1]] = []
            elif 'from GR_JJS NUM' in real_tp_str or 'IN GR_JJS NUM' in real_tp_str:
                idx = real_tp_str.index('from GR_JJS NUM') if 'from GR_JJS NUM' in real_tp_str else real_tp_str.index('IN GR_JJS NUM')
                left_str = real_tp_str[:idx]
                idx = left_str.count(" ")
                sq.question_tokens[list_idx[i]][real_tp_list[idx+1][1]] = SToken(text="descending",lemma="descend")
                sq.question_tokens[list_idx[i]][real_tp_list[idx+2][1]] = SToken(text="order",lemma="order")
                sq.col_match[list_idx[i]][real_tp_list[idx+1][1]] = []
                sq.col_match[list_idx[i]][real_tp_list[idx+2][1]] = []
            continue
        elif ('IN SM_JJS COL NUM' in real_tp_str or "from SM_JJS SM_JJS NUM" in real_tp_str or "from SM_JJS NUM" in real_tp_str  or 'IN SM_JJS SM_JJS NUM' in real_tp_str  or 'IN SM_JJS NUM' in real_tp_str) and ("first" in question[i] or "one" in question[i]):
            if 'IN SM_JJS COL NUM' in real_tp_str:
                idx = real_tp_str.index('IN SM_JJS COL NUM')
                left_str = real_tp_str[:idx]
                idx = left_str.count(" ")
                sq.question_tokens[list_idx[i]][real_tp_list[idx+1][1]] = SToken(text="ascending",lemma="ascend")
                sq.question_tokens[list_idx[i]][real_tp_list[idx+3][1]] = SToken(text="order",lemma="order")
                sq.col_match[list_idx[i]][real_tp_list[idx+1][1]] = []
                sq.col_match[list_idx[i]][real_tp_list[idx+3][1]] = []
            elif 'from SM_JJS SM_JJS NUM' in real_tp_str or 'IN SM_JJS SM_JJS NUM' in real_tp_str:
                idx = real_tp_str.index('from SM_JJS SM_JJS NUM')  if 'from SM_JJS SM_JJS NUM' in real_tp_str else real_tp_str.index('IN SM_JJS SM_JJS NUM')
                left_str = real_tp_str[:idx]
                idx = left_str.count(" ")
                sq.question_tokens[list_idx[i]][real_tp_list[idx+1][1]] = SToken(text="ascending",lemma="ascend")
                sq.question_tokens[list_idx[i]][real_tp_list[idx+2][1]] = SToken(text="to",lemma="to")
                sq.question_tokens[list_idx[i]][real_tp_list[idx+3][1]] = SToken(text="end",lemma="end")
                sq.col_match[list_idx[i]][real_tp_list[idx+1][1]] = []
                sq.col_match[list_idx[i]][real_tp_list[idx+2][1]] = []
                sq.col_match[list_idx[i]][real_tp_list[idx+3][1]] = []
            elif 'from SM_JJS NUM' in real_tp_str or 'IN SM_JJS NUM' in real_tp_str:
                idx = real_tp_str.index('from SM_JJS NUM') if 'from SM_JJS NUM' in real_tp_str else real_tp_str.index('IN SM_JJS NUM')
                left_str = real_tp_str[:idx]
                idx = left_str.count(" ")
                sq.question_tokens[list_idx[i]][real_tp_list[idx+1][1]] = SToken(text="ascending",lemma="ascend")
                sq.question_tokens[list_idx[i]][real_tp_list[idx+2][1]] = SToken(text="order",lemma="order")
                sq.col_match[list_idx[i]][real_tp_list[idx+1][1]] = []
                sq.col_match[list_idx[i]][real_tp_list[idx+2][1]] = []
            continue
        elif "from " in real_tp_str and (' SM_JJS GR_JJS' in real_tp_str or ' SM_GRSM GR_GRSM' in real_tp_str or ' SM_JJS GR_GRSM' in real_tp_str or ' SM_GRSM GR_JJS' in real_tp_str):
            if ' SM_JJS GR_JJS' in real_tp_str:
                idx = real_tp_str.index(' SM_JJS GR_JJS')
            elif ' SM_GRSM GR_GRSM' in real_tp_str:
                idx = real_tp_str.index(' SM_GRSM GR_GRSM')
            elif ' SM_JJS GR_GRSM' in real_tp_str:
                idx = real_tp_str.index(' SM_JJS GR_GRSM')
            elif ' SM_GRSM GR_JJS' in real_tp_str:
                idx = real_tp_str.index(' SM_GRSM GR_JJS')
            else:
                continue
            left_str = real_tp_str[:idx]
            idx = left_str.count(" ") + 1
            sq.question_tokens[list_idx[i]][real_tp_list[idx][1]] = SToken(text="ascending",lemma="ascend")
            sq.question_tokens[list_idx[i]][real_tp_list[idx+1][1]] = SToken(text="end",lemma="end")
            sq.col_match[list_idx[i]][real_tp_list[idx][1]] = []
            sq.col_match[list_idx[i]][real_tp_list[idx+1][1]] = []

        elif "from " in real_tp_str and (' GR_JJS SM_JJS' in real_tp_str or ' GR_GRSM SM_GRSM' in real_tp_str or ' GR_JJS SM_GRSM' in real_tp_str or ' GR_GRSM SM_JJS' in real_tp_str):
            if ' GR_JJS SM_JJS' in real_tp_str:
                idx = real_tp_str.index(' GR_JJS SM_JJS')
            elif ' GR_GRSM SM_GRSM' in real_tp_str:
                idx = real_tp_str.index(' GR_GRSM SM_GRSM')
            elif ' GR_JJS SM_GRSM' in real_tp_str:
                idx = real_tp_str.index(' GR_JJS SM_GRSM')
            elif ' GR_GRSM SM_JJS' in real_tp_str:
                idx = real_tp_str.index(' GR_GRSM SM_JJS')
            else:
                continue
            left_str = real_tp_str[:idx]
            idx = left_str.count(" ") + 1
            sq.question_tokens[list_idx[i]][real_tp_list[idx][1]] = SToken(text="descending",lemma="descend")
            sq.question_tokens[list_idx[i]][real_tp_list[idx+1][1]] = SToken(text="end",lemma="end")
            sq.col_match[list_idx[i]][real_tp_list[idx][1]] = []
            sq.col_match[list_idx[i]][real_tp_list[idx+1][1]] = []
        if ' first ' in question[i] and not re.fullmatch(r'.*?\sfirst\s(([a-z]*|,)\s){0,4}name(s){0,1}\s.*', question[i]):
            idx_first = -1
            for k,t_idx in enumerate(real_tp_list):
                if t_idx[0] == "NUM" and sq.question_tokens[list_idx[i]][t_idx[1]].text == "first":
                    idx_first = k
                    break
            if idx_first >= 0:
                sq,yes = reset_frist_last_col_sq(sq,idx_first,list_idx,i,real_tp_list,schema,["COL","STC","SC","TABLE-COL"],"time","earliest","early")
                if yes:
                    continue
                sq,yes = reset_frist_last_tb_sq(sq,idx_first,list_idx,i,real_tp_list,schema,["ST","TABLE","STC","TABLE-COL"],"earliest","early")
                if yes:
                    continue
                sq,yes = reset_frist_last_col_sq(sq,idx_first,list_idx,i,real_tp_list,schema,["COL","STC","SC","TABLE-COL"],"number","1","1")
                if yes:
                    continue
        if ' last ' in question[i] and not re.fullmatch(r'.*?\slast\s(([a-z]*|,)\s){0,3}name(s){0,1}\s.*', question[i]):
            idx_first = -1
            for k,t_idx in enumerate(real_tp_list):
                if t_idx[0] in ["last","DATE","YEAR"] and sq.question_tokens[list_idx[i]][t_idx[1]].text == "last":
                    idx_first = k
                    break
            if idx_first >= 0:
                sq,yes = reset_frist_last_col_sq(sq,idx_first,list_idx,i,real_tp_list,schema,["COL","STC","SC","TABLE-COL"],"time","latest","late")
                if yes:
                    continue
                sq,yes = reset_frist_last_tb_sq(sq,idx_first,list_idx,i,real_tp_list,schema,["ST","TABLE","STC","TABLE-COL"],"latest","late")
                if yes:
                    continue



        for_break = False
        if re.fullmatch(r".*(start|end)(.){0,5}with.*", question[i]) and "SM_JJS" not in tps and "GR_JJS" not in tps:
            if " tilt " in question[i]:
                continue
            target = ["start","end"]
            for j,tok in zip(range(len(sq.question_tokens[list_idx[i]])-1,-1,-1),reversed(sq.question_tokens[list_idx[i]])):
                for t in target:
                    if t in tok.text:
                        if tok.text.endswith("ing"):
                            sq.question_tokens[list_idx[i]][j] = SToken(text="tilting")
                        elif tok.text.endswith("ed"):
                            sq.question_tokens[list_idx[i]][j] = SToken(text="tilted")
                        else:
                            sq.question_tokens[list_idx[i]][j] = SToken(text="tilt")
                        sq.col_match[list_idx[i]][j] = []
                        for_break = True
                        break
                if for_break:
                    break
        elif re.fullmatch(r"(.*?(contain|includ|with\s|without\s|have\s|has\s|having\s|had\s|like|\sas\s)|as\s).*(substring|string|texts\s|text\s|char\s|chars\s|character\s|letter|word|phrase|prefix|suffix).*", question[i]) or re.fullmatch(r".*(contain|includ|like)(.){0,4}(').*?(').*", question[i]) or re.fullmatch(r".*(hav|includ|has|with)(.){0,4}((an|a|the)\s){0,1}(').*?(')\sin(\s|\?|\.).*", question[i]):
            if " tilt " in question[i]:
                continue
            target = ["contain","includ","like","with","have","has","having","had","as","without"]
            for j,tok in zip(range(len(sq.question_tokens[list_idx[i]])-1,-1,-1),reversed(sq.question_tokens[list_idx[i]])):
                for t in target:
                    if (tok.text.startswith(t) and tok.text!="without") or t == tok.text :
                        if t == "without" and j + 1 < len(sq.question_tokens[list_idx[i]]):
                            sq.question_tokens[list_idx[i]][j] = SToken(text="not")
                            sq.col_match[list_idx[i]][j] = []
                            j += 1

                        if tok.text.endswith("ing"):
                            sq.question_tokens[list_idx[i]][j] = SToken(text="tilting")
                        elif tok.text.endswith("ed"):
                            sq.question_tokens[list_idx[i]][j] = SToken(text="tilted")
                        else:
                            sq.question_tokens[list_idx[i]][j] = SToken(text="tilt")
                        sq.col_match[list_idx[i]][j] = []
                        for_break = True
                        break
                if for_break:
                    break
        elif re.fullmatch(r".*(with|contain|contains|containing|contained|include|including|includes|included|includ|includs|have|had|has|having|like|likes)\s([A-Z]){1,10}(\s|\.|\?){0,1}.*", question[i]):
            if " tilt " in question[i]:
                continue
            target = ["contain","includ","with","have","has","having","had","like", "likes"]
            for j,tok in zip(range(len(sq.question_tokens[list_idx[i]])-1,-1,-1),reversed(sq.question_tokens[list_idx[i]])):
                for t in target:
                    if tok.text.startswith(t):# if t in tok.text:
                        if j+1 < len(tps) and tps[j+1] in ["SDB","DB","PDB","UDB"] and t in ["with","have","has","having","had"]:
                            break
                        if j+1 < len(tps) and tps[j+1] in ["COL","STC","SC","TABLE-COL","TABLE","ST"]:
                            break
                        elif j+1 < len(tps) and not sq.question_tokens[list_idx[i]][j+1].text.islower():
                            if tok.text.endswith("ing"):
                                sq.question_tokens[list_idx[i]][j] = SToken(text="tilting")
                            elif tok.text.endswith("ed"):
                                sq.question_tokens[list_idx[i]][j] = SToken(text="tilted")
                            else:
                                sq.question_tokens[list_idx[i]][j] = SToken(text="tilt")
                            sq.col_match[list_idx[i]][j] = [] 
                            for_break = True
                            break
                if for_break:
                    break   
        elif "DB" in real_tp_str and ( real_tp_str.endswith("TABLE PDB") or real_tp_str.endswith("TABLE PDB PDB") or real_tp_str.endswith("TABLE PDB PDB PDB") or real_tp_str.endswith("TABLE PDB PDB PDB PDB") or real_tp_str.endswith("TABLE PDB PDB PDB PDB PDB") or real_tp_str.endswith("TABLE SDB") or real_tp_str.endswith("TABLE SDB SDB") or real_tp_str.endswith("TABLE SDB SDB SDB") or real_tp_str.endswith("TABLE SDB SDB SDB SDB") or real_tp_str.endswith("TABLE SDB SDB SDB SDB SDB") or real_tp_str.endswith("TABLE UDB") or real_tp_str.endswith("TABLE UDB UDB") or real_tp_str.endswith("TABLE UDB UDB UDB") or real_tp_str.endswith("TABLE UDB UDB UDB UDB") or real_tp_str.endswith("TABLE UDB UDB UDB UDB UDB")\
            or real_tp_str.endswith("TABLE PDB NN") or real_tp_str.endswith("TABLE PDB PDB NN") or real_tp_str.endswith("TABLE PDB PDB PDB NN") or real_tp_str.endswith("TABLE PDB PDB PDB PDB NN") or real_tp_str.endswith("TABLE PDB PDB PDB PDB PDB NN") or real_tp_str.endswith("TABLE SDB NN") or real_tp_str.endswith("TABLE SDB SDB NN") or real_tp_str.endswith("TABLE SDB SDB SDB NN") or real_tp_str.endswith("TABLE SDB SDB SDB SDB NN") or real_tp_str.endswith("TABLE SDB SDB SDB SDB SDB NN") or real_tp_str.endswith("TABLE UDB NN") or real_tp_str.endswith("TABLE UDB UDB NN") or real_tp_str.endswith("TABLE UDB UDB UDB NN") or real_tp_str.endswith("TABLE UDB UDB UDB UDB NN") or real_tp_str.endswith("TABLE UDB UDB UDB UDB UDB NN")\
            or real_tp_str.endswith("TABLE * PDB NN") or real_tp_str.endswith("TABLE * PDB PDB NN") or real_tp_str.endswith("TABLE * PDB PDB PDB NN") or real_tp_str.endswith("TABLE * PDB PDB PDB PDB NN") or real_tp_str.endswith("TABLE * PDB PDB PDB PDB PDB NN") or real_tp_str.endswith("TABLE * SDB NN") or real_tp_str.endswith("TABLE * SDB SDB NN") or real_tp_str.endswith("TABLE * SDB SDB SDB NN") or real_tp_str.endswith("TABLE * SDB SDB SDB SDB NN") or real_tp_str.endswith("TABLE * SDB SDB SDB SDB SDB NN") or real_tp_str.endswith("TABLE * UDB NN") or real_tp_str.endswith("TABLE * UDB UDB NN") or real_tp_str.endswith("TABLE * UDB UDB UDB NN") or real_tp_str.endswith("TABLE * UDB UDB UDB UDB NN") or real_tp_str.endswith("TABLE * UDB UDB UDB UDB UDB NN")\
            or real_tp_str.endswith("TABLE PDB *") or real_tp_str.endswith("TABLE PDB PDB *") or real_tp_str.endswith("TABLE PDB PDB PDB *") or real_tp_str.endswith("TABLE PDB PDB PDB PDB *") or real_tp_str.endswith("TABLE PDB PDB PDB PDB PDB *") or real_tp_str.endswith("TABLE SDB *") or real_tp_str.endswith("TABLE SDB SDB *") or real_tp_str.endswith("TABLE SDB SDB SDB *") or real_tp_str.endswith("TABLE SDB SDB SDB SDB *") or real_tp_str.endswith("TABLE SDB SDB SDB SDB SDB *") or real_tp_str.endswith("TABLE UDB *") or real_tp_str.endswith("TABLE UDB UDB *") or real_tp_str.endswith("TABLE UDB UDB UDB *") or real_tp_str.endswith("TABLE UDB UDB UDB UDB *") or real_tp_str.endswith("TABLE UDB UDB UDB UDB UDB *")\
            or real_tp_str.endswith("PDB TABLE") or real_tp_str.endswith("UDB TABLE") or real_tp_str.endswith("SDB TABLE") ):
            if real_tp_str.endswith("PDB TABLE") or real_tp_str.endswith("UDB TABLE") or real_tp_str.endswith("SDB TABLE") :
                table_idx = real_tp_list[-1][1]
                pdb_idx = real_tp_list[-2][1]
                num_pdb = 1
            else:
                if "TABLE PDB" in real_tp_str or "TABLE * PDB" in real_tp_str:
                    num_pdb = real_tp_str.count("PDB")
                elif "TABLE SDB" in real_tp_str  or "TABLE * SDB" in real_tp_str:
                    num_pdb = real_tp_str.count("SDB")   
                elif "TABLE UDB" in real_tp_str  or "TABLE * UDB" in real_tp_str:
                    num_pdb = real_tp_str.count("UDB") 
                if real_tp_str.endswith("NN") or real_tp_str.endswith("*"):
                    if "TABLE * " in real_tp_str:
                        table_idx = real_tp_list[-num_pdb-3][1]
                    else:
                        table_idx = real_tp_list[-num_pdb-2][1]
                    pdb_idx = real_tp_list[-num_pdb-1][1]
                else:
                    table_idx = real_tp_list[-num_pdb-1][1]
                    pdb_idx = real_tp_list[-num_pdb][1]
            count_name = 0
            store_pdb_idx = pdb_idx
            for tb in sq.table_match[list_idx[i]][table_idx]:
                for k,cc in enumerate(schema.tbl_col_tokens_lemma_str[tb]):
                    if cc.endswith(" name") or cc.endswith(" title") or cc == "name" or cc == "title":
                        for dbc in range(num_pdb):
                            sq.db_match[list_idx[i]][pdb_idx].append([schema.tbl_col_idx_back[tb][k],[pdb_idx,pdb_idx]])
                            sq.pattern_tok[list_idx[i]][pdb_idx] = "DB"
                            pdb_idx += 1
                        pdb_idx = store_pdb_idx
        elif "NUM" in real_tp_str and ( real_tp_str.endswith("ST NUM") or real_tp_str.endswith("TABLE NUM") ):
            num_pdb = real_tp_str.count("NUM")
            table_idx = real_tp_list[-num_pdb-1][1]
            
            for jj in range(len(sq.table_match)):
                for kk in range(len(sq.table_match[jj])):
                    for tb in sq.table_match[jj][kk]:
                        if tb:
                            sq = add_name_col(sq,schema,tb,list_idx[i],table_idx,[sq.question_tokens[list_idx[i]][table_idx].lemma_+" id"],table_prefix=schema.table_tokens_lemma_str[tb].split(" | ")[0])

        
        if re.fullmatch(r"(.*?\s(not|no)|not|no)\s(has\s|have\s|had\s|having\s){0,1}[a-z]{1,10}\sthan\s.*", question[i]) and tp_str.count("GRSM") == 1 and tp_str.count("GR_GRSM") == 1 and tp_str.count("NOT") == 1:    
            sq.question_tokens[list_idx[i]][tps.index("NOT")] = SToken(text="the")
            sq.question_tokens[list_idx[i]][tps.index("GR_GRSM")] = SToken(text="uglier")
    return sq



def not_col_sub_q(sq,schema):
    for i,cols in enumerate(sq.col_match):
        if cols.count([]) == len(cols) and  i >= 1 and sq.table_match[i-1][-1] and ("PDB" in sq.pattern_tok[i] or "SDB" in sq.pattern_tok[i] or "UDB" in sq.pattern_tok[i]):
            for tb in sq.table_match[i-1][-1]:
                for j,ptok in enumerate(sq.pattern_tok[i]):
                    if "DB" in ptok:
                        sq = add_name_col(sq,schema,tb,i,j)
    for (k,qts),dbs in zip(enumerate(sq.question_tokens),sq.db_match):
        for (i,qt),db in zip(enumerate(qts),dbs):
            if db and str_is_date(qt.text,qts,i):
                delete_db = False
                for j in range(i-1,-1,-1):
                    if qts[j].text in ["before","after"]:
                        delete_db = True
                        break
                    elif qts[j].text in ["and","or",",","in","on","at","equal"]:
                        break
                    elif qts[j].text in S_ADJ_WORD_DIRECTION or qts[j].text in ABSOLUTELY_GRSM_DICT:
                        break
                if delete_db:
                    for j in range(db[0][1][1]-db[0][1][0]+1):
                        for dbdb in sq.db_match[k][i+j]:
                            if not sq.col_match[k][i+j]:
                                sq.col_match[k][i+j] = [[dbdb[0]],[1],[1],[[dbdb[0]]]]
                            else:
                                sq.col_match[k][i+j][0].append(dbdb[0])
                                sq.col_match[k][i+j][1].append(1)
                                sq.col_match[k][i+j][2].append(1)
                                sq.col_match[k][i+j][3].append(dbdb[0])
                        sq.db_match[k][i+j] = []
    return sq
            



def sgrsm_add_col_match(token_pattern,question,list_idx,sq,schema):
    for i,tps in enumerate(token_pattern):
        for j,tp in enumerate(tps):
            if "SJJS" in tp or "SGRSM" in tp:

                # SJJS TABLE:
                if j + 1 < len(tps) and tps[j+1] == "TABLE":
                    result = []
                    sgrsm_word = None
                    for t in sq.table_match[list_idx[i]][j+1]:
                        re_list_idx,re_list_dir = sjjs_table(sq.question_tokens[list_idx[i]][j], schema.tbl_col_tokens_lemma_str[t], schema.tbl_col_idx_back[t])
                        if not sgrsm_word or sgrsm_word == re_list_dir:
                            result.extend(re_list_idx)
                            sgrsm_word = re_list_dir
                        elif sgrsm_word and re_list_dir and sgrsm_word != re_list_dir:
                            sgrsm_word = "ERROR"
                            break
                    if sgrsm_word and result and sgrsm_word != "ERROR":
                        if not sq.col_match[list_idx[i]][j]:
                            sq.col_match[list_idx[i]][j] = [[],[],[],[]]
                        for col in result:
                            sq.col_match[list_idx[i]][j][0].append(col)
                            sq.col_match[list_idx[i]][j][1].append(1)
                            sq.col_match[list_idx[i]][j][2].append(1)
                            sq.col_match[list_idx[i]][j][3].append(col)         
                        continue

                # FIND COL FROM QUESTION
                result = is_there_sgrsm_and_gr_or_sm(sq.question_tokens[list_idx[i]],sq.question_tokens[list_idx[i]][j],j)
                if not sq.col_match[list_idx[i]][j] and (not result or sq.col_match[list_idx[i]].count([]) == len(sq.col_match[list_idx[i]])):
                    table_idxs = []
                    for t in sq.table_match[list_idx[i]]:
                        table_idxs.extend(t)
                    if not table_idxs:
                        for ts in sq.table_match:
                            for t in ts:
                                table_idxs.extend(t)
                    if not table_idxs:
                        table_idxs = [-1]
                    agg_id, col_id, sgrsm_word = get_AWD_column(sq.question_tokens[list_idx[i]][j].lemma_, table_idxs, schema, restrict=True, re_all_word = True)
                    if agg_id < 0 and -1 not in table_idxs:
                        agg_id, col_id, sgrsm_word = get_AWD_column(sq.question_tokens[list_idx[i]][j].lemma_, [-1], schema, restrict=True, re_all_word = True)
                    if agg_id >= 0:
                        sq.col_match[list_idx[i]][j] = [[],[],[],[]]
                        if type(col_id) == list:
                            for col in col_id:
                                sq.col_match[list_idx[i]][j][0].append(col)
                                sq.col_match[list_idx[i]][j][1].append(1)
                                sq.col_match[list_idx[i]][j][2].append(1)
                                sq.col_match[list_idx[i]][j][3].append(col)
                        else:
                            for col in col_id:
                                sq.col_match[list_idx[i]][j][0].append(col_id)
                                sq.col_match[list_idx[i]][j][1].append(1)
                                sq.col_match[list_idx[i]][j][2].append(1)
                                sq.col_match[list_idx[i]][j][3].append(col_id)
    return sq,token_pattern


def add_name_col(sq,schema,tb_idx,sq_idx,added_col_idx,words = [" name"," title","name","title"],table_prefix=None,pk=False):
        def check_words_match(col_w,words):
            if len(words) == 4:
                for cw in col_w.split(" | "):
                    if cw.endswith(words[0]) or cw.endswith(words[1]) or cw == words[2] or cw == words[3]:
                        return True
            else:
                for cw in col_w.split(" | "):
                    for w in words:
                        if w[0] == " " and cw.endswith(w):
                            return True
                        elif cw == w:
                            return True
            return False
        for k,cc in enumerate(schema.tbl_col_tokens_lemma_str[tb_idx]):
            if check_words_match(cc,words) or (table_prefix and check_words_match(table_prefix+" " + cc,words)) or (pk and schema.tbl_col_idx_back[tb_idx][k] in schema.primaryKey ):
                if sq.col_match[sq_idx][added_col_idx] == []:
                    sq.col_match[sq_idx][added_col_idx] = [[schema.tbl_col_idx_back[tb_idx][k]],[1],[1],[schema.tbl_col_idx_back[tb_idx][k]]]
                else:
                    if schema.tbl_col_idx_back[tb_idx][k] not in sq.col_match[sq_idx][added_col_idx][0]:
                        sq.col_match[sq_idx][added_col_idx][0].append(schema.tbl_col_idx_back[tb_idx][k])
                        sq.col_match[sq_idx][added_col_idx][1].append(1)
                        sq.col_match[sq_idx][added_col_idx][2].append(1)
                        sq.col_match[sq_idx][added_col_idx][3].append(schema.tbl_col_idx_back[tb_idx][k])
        return sq


def select_col_and_col_of(token_pattern,question,list_idx,sq,schema,select_num):
    def sub_tok_match(str1,str2):
        for s in str1.split(" "):
            if s in str2:
                return True
        return False
    def col_not_in_tables(cols,tables,schema):
        if cols:
            for c in cols[0]:
                if schema.column_names_original[c][0] in tables:
                    return False
        return True

    
    for i,tps in enumerate(token_pattern):
        tp_str2 = ""
        index_list = []
        for j, tp, q in zip(range(len(tps)), tps, question[i].lower().split(" ")):
            if tp in ["#","?","WP","NN"]:
                if q not in ["of","and","who","where","when","how"]:
                    continue
                tp_str2 += (q+" ")
            else:
                tp_str2 += (tp+" ")
            index_list.append(j)
        patterns = ["COL and COL COL of COL", "COL and COL of COL", "COL and COL COL of TABLE", "COL and COL of TABLE",\
        "COL and COL COL of all COL", "COL and COL of all COL", "COL and COL COL of all TABLE", "COL and COL of all TABLE",\
            "COL and COL COL IN COL","COL and COL IN COL","COL and COL IN TABLE","COL and COL COL IN TABLE",\
            "COL and COL COL IN all COL","COL and COL IN all COL","COL and COL IN all TABLE","COL and COL COL IN all TABLE"]
        for p in patterns:
            if p in tp_str2 and tp_str2.count(p) == 1:
                idx = tp_str2.index(p)
                lfidx = tp_str2[0:idx].count(" ")
                ridx = index_list[lfidx + p.count(" ")]
                lfidx = index_list[lfidx] 
                if lfidx>0 and tps[lfidx-1] in ["TABLE","TABLE-COL"]:
                    ridx = lfidx-1
                    tables = sq.table_match[list_idx[i]][ridx]
                elif lfidx>1 and sq.question_tokens[list_idx[i]][lfidx-1].text == "with" and tps[lfidx-2] in ["TABLE","TABLE-COL"]:
                    ridx = lfidx-2
                    tables = sq.table_match[list_idx[i]][ridx]
                if ridx + 1 < len(tps) and tps[ridx+1] == "TABLE":
                    ridx += 1
                    tables = sq.table_match[list_idx[i]][ridx]
                elif tps[ridx] == "TABLE":
                    tables = sq.table_match[list_idx[i]][ridx]
                else:
                    tables = look_for_table_idx(sq, list_idx[i], 1, schema)
                if (sq.col_match[list_idx[i]][lfidx+2] and col_not_in_tables(sq.col_match[list_idx[i]][lfidx+2],tables,schema)) or (sq.col_match[list_idx[i]][lfidx+3] and col_not_in_tables(sq.col_match[list_idx[i]][lfidx+3],tables,schema)):
                    break
                ts = TokenString(None,[sq.question_tokens[list_idx[i]][ridx],sq.question_tokens[list_idx[i]][lfidx]])
                full_match = col_match_main(tables,ts,schema,[sq.table_match[list_idx[i]][ridx],sq.table_match[list_idx[i]][lfidx]],True)
                if full_match[1] and len(full_match[1][0]) > 1 and max(full_match[1][1]) == 2:
                    for f in range(len(full_match[1][0])-1,-1,-1):
                        if full_match[1][1][i] != 2 and not sub_tok_match(schema.table_tokens_lemma_str[schema.column_tokens_table_idx[full_match[1][0][i]]],ts.lemma_ ) :
                            del full_match[1][0][i]
                            del full_match[1][1][i]
                            del full_match[1][2][i]
                            del full_match[1][3][i]
                if not sq.col_match[list_idx[i]][lfidx] or (full_match[1] and max(full_match[1][1]) >= max(sq.col_match[list_idx[i]][lfidx][1])):
                    sq.col_match[list_idx[i]][lfidx] = full_match[1]
                if full_match[0]:
                    if sq.col_match[list_idx[i]][ridx] == []:
                        pass
                    else:
                        for f in range(len(full_match[0][0])):
                            if full_match[0][0][f] not in sq.col_match[list_idx[i]][ridx][0]:
                                sq.col_match[list_idx[i]][ridx][0].append(full_match[0][0][f])
                                sq.col_match[list_idx[i]][ridx][1].append(full_match[0][1][f])
                                sq.col_match[list_idx[i]][ridx][2].append(full_match[0][2][f])
                                sq.col_match[list_idx[i]][ridx][3].append(full_match[0][3][f])
                break
        
        already_add = False
        patterns = ["COL of TABLE and TABLE ","COL of TABLE and JJ TABLE ","COL of TABLE and of TABLE ",\
            "COL of TABLE and TABLE ? ","COL of TABLE and JJ TABLE ? ","COL of TABLE and of TABLE ? ",\
            "COL of TABLE and TABLE . ","COL of TABLE and JJ TABLE . ","COL of TABLE and of TABLE . " \
            ,"TABLE and TABLE COL ","TABLE and COL COL "]
        patterns_ = ["COL of TABLE and TABLE IN TABLE-COL ","COL of TABLE and TABLE IN TABLE ",] 
        for p in (patterns+patterns_):
            if tp_str2.endswith(p) and tp_str2.count(p) == 1:
                if p in patterns_:
                    p = p.split(" ")
                    p = " ".join(p[:-3]) + " "
                idx = tp_str2.index(p)
                lfidx = tp_str2[0:idx].count(" ")
                if p.endswith(" ? ") or p.endswith(" . "):
                    ridx = index_list[lfidx + p.count(" ") - 2]
                else:
                    ridx = index_list[lfidx + p.count(" ") - 1]
                lfidx = index_list[lfidx] 

                (ridx,lfidx) = (lfidx,ridx) if p.startswith("TABLE ") else (ridx,lfidx)

                if tps[ridx] == "TABLE":
                    tables = sq.table_match[list_idx[i]][ridx]
                else:
                    tables = look_for_table_idx(sq, list_idx[i], 1, schema)
                ts = TokenString(None,[sq.question_tokens[list_idx[i]][ridx],sq.question_tokens[list_idx[i]][lfidx]])
                full_match = col_match_main(tables,ts,schema,[sq.table_match[list_idx[i]][ridx],sq.table_match[list_idx[i]][lfidx]],True)
                if full_match[1]:
                    if sq.col_match[list_idx[i]][ridx] == []:
                        sq.col_match[list_idx[i]][ridx] = full_match[1]
                    else:
                        for f in range(len(full_match[0][0])):
                            if full_match[0][0][f] not in sq.col_match[list_idx[i]][ridx][0]:
                                sq.col_match[list_idx[i]][ridx][0].append(full_match[0][0][f])
                                sq.col_match[list_idx[i]][ridx][1].append(full_match[0][1][f])
                                sq.col_match[list_idx[i]][ridx][2].append(full_match[0][2][f])
                                sq.col_match[list_idx[i]][ridx][3].append(full_match[0][3][f])
                already_add = True

        if already_add:
            continue

        patterns1 = ["who GR_SJJS ","who SM_SJJS ","who * GR_SJJS ","who * SM_SJJS "]
        patterns2 = ["where ",]
        patterns3 = ["when ",]
        patterns5 = ["* TABLE DB ",]
        patterns4 = patterns1 + patterns2 + patterns3 + patterns5
        for p in patterns4:
            if tp_str2.startswith(p) and tp_str2.count(p) == 1:
                if p in patterns5:
                    if select_num == 1 and len(token_pattern) == 1:
                        idx_start = index_list[1]
                    else:
                        break
                else:
                    idx_start = index_list[0]
                for tbs in sq.table_match[list_idx[i]]:
                    for tb in tbs:
                        if p in patterns2:
                            sq = add_name_col(sq,schema,tb,list_idx[i],idx_start,[" address"," location","address","location"])
                        elif p in patterns3:
                            sq = add_name_col(sq,schema,tb,list_idx[i],idx_start,["date from"," date to"," date","date","year"," year"])
                        else:
                            sq = add_name_col(sq,schema,tb,list_idx[i],idx_start)
                        
                if not sq.col_match[list_idx[i]][idx_start]:
                    if p in patterns2:
                        sq = add_name_col(sq,schema,-1,list_idx[i],idx_start,[" address"," location","address","location"])
                    elif p in patterns3:
                        sq = add_name_col(sq,schema,-1,list_idx[i],idx_start,["date from"," date to"," date","date","year"," year"])
                    else:
                        sq = add_name_col(sq,schema,-1,list_idx[i],idx_start)
                already_add = True

        if already_add:
            continue

        patterns1 = ["TABLE of COL COL ? ","TABLE of COL COL . ","TABLE of COL COL ","TABLE of COL ? ","TABLE of COL . ","TABLE of COL ","TABLE and COL ","TABLE and COL COL "]
        patterns2 = ["and TABLE ","and TABLE . ","and TABLE ? ","GR_JJS TABLE-COL ","SM_JJS TABLE-COL "]
        patterns22 = ["TABLE ","TABLE-COL ","TABLE TABLE-COL ","TABLE-COL TABLE-COL "]
        patterns3 = patterns1 + patterns2 + patterns22
        for p in patterns3:
            if p in patterns1 and not (tp_str2.count("TABLE") == p.count("TABLE") and tp_str2.count("COL") == p.count("COL")):
                continue
            elif p in patterns22 and (len(list_idx) == 2 or tp_str2.count(" ") - p.count(" ") > 1 or tp_str2.startswith("how TABLE ") or tp_str2.endswith("how TABLE-COL ")  or "AGG" in tp_str2 or "JJS" in tp_str2 ):
                continue
            if tp_str2.endswith(p) and tp_str2.count(p) == 1:
                idx = tp_str2.index(p)
                lfidx = tp_str2[0:idx].count(" ")
                if p.endswith(" ? ") or p.endswith(" . "):
                    ridx = index_list[lfidx + p.count(" ") - 2]
                else:
                    ridx = index_list[lfidx + p.count(" ") - 1]
                lfidx = index_list[lfidx] 

                if p in patterns2 or p in patterns22:
                    add_name = True
                    (ridx,lfidx) = (lfidx,ridx)
                    if p == "TABLE TABLE-COL " or p == "TABLE-COL TABLE-COL ":
                        lfidx -= 1
                else:
                    add_name = False
                    for tb in sq.table_match[list_idx[i]][lfidx]:
                        all_not_in = True
                        for col in sq.col_match[list_idx[i]][ridx][0]:
                            if schema.column_names_original[col][0] == tb:
                                all_not_in = False
                        if all_not_in:
                            add_name = True
                if add_name:
                    sq = add_fk_to_table(sq,schema,list_idx[i],lfidx)
                    old_col_match = copy.deepcopy(sq.col_match[list_idx[i]][lfidx])
                    for tb in sq.table_match[list_idx[i]][lfidx]:
                        sq = add_name_col(sq,schema,tb,list_idx[i],lfidx,[sq.question_tokens[list_idx[i]][lfidx].lemma_+" name"],table_prefix=schema.table_tokens_lemma_str[tb].split(" | ")[0])
                    if old_col_match == sq.col_match[list_idx[i]][lfidx]:
                        for tb in sq.table_match[list_idx[i]][lfidx]:
                            sq = add_name_col(sq,schema,tb,list_idx[i],lfidx,[sq.question_tokens[list_idx[i]][lfidx].lemma_+" detail",sq.question_tokens[list_idx[i]][lfidx].lemma_+" description"],table_prefix=schema.table_tokens_lemma_str[tb].split(" | ")[0])
                    if old_col_match == sq.col_match[list_idx[i]][lfidx]:
                        for tb in sq.table_match[list_idx[i]][lfidx]:
                            sq = add_name_col(sq,schema,tb,list_idx[i],lfidx,[sq.question_tokens[list_idx[i]][lfidx].lemma_+" content"],table_prefix=schema.table_tokens_lemma_str[tb].split(" | ")[0])
                    if old_col_match == sq.col_match[list_idx[i]][lfidx]:
                        for tb in sq.table_match[list_idx[i]][lfidx]:
                            sq = add_name_col(sq,schema,tb,list_idx[i],lfidx,[sq.question_tokens[list_idx[i]][lfidx].lemma_+" id"],table_prefix=schema.table_tokens_lemma_str[tb].split(" | ")[0])
        if already_add:
            continue

        patterns1 = ["how TABLE TABLE ","how TABLE "]
        patterns2 = ["how JJ TABLE ","how JJ * TABLE ","how JJ * * TABLE "]
        patterns3 = ["how JJ TABLE-COL ","how JJ * TABLE-COL ","how JJ * * TABLE-COL "]
        patterns4 = ["* AGG of JJ TABLE ","* AGG of JJ TABLE-COL "]

        patterns5 = patterns1 + patterns2 + patterns3 + patterns4
        for p in patterns5:
            if tp_str2.startswith(p) and tp_str2.count(p) == 1:
                idx = tp_str2.index(p)
                lfidx = tp_str2[0:idx].count(" ") + 1
                if (p in patterns2 or p in patterns3) and ("how many different " in question[i].lower() or "how many distinct " in question[i].lower()):
                    if p in patterns2:
                        tb_idx = tp_str2.split(" ").index("TABLE")
                    else:
                        tb_idx = tp_str2.split(" ").index("TABLE-COL")

                    lfidx = index_list[lfidx+1] 
                    for tb in sq.table_match[list_idx[i]][index_list[tb_idx]]:
                        sq = add_name_col(sq,schema,tb,list_idx[i],lfidx,[sq.question_tokens[list_idx[i]][lfidx].lemma_+" id"],table_prefix=schema.table_tokens_lemma_str[tb].split(" | ")[0],pk=True)
                elif (p in patterns4) and ("number of distinct " in question[i].lower() or "number of different " in question[i].lower()):
                    if "TABLE " in p:
                        tb_idx = tp_str2.split(" ").index("TABLE")
                    else:
                        tb_idx = tp_str2.split(" ").index("TABLE-COL")

                    lfidx = index_list[lfidx+1] 
                    for tb in sq.table_match[list_idx[i]][index_list[tb_idx]]:
                        sq = add_name_col(sq,schema,tb,list_idx[i],lfidx,[sq.question_tokens[list_idx[i]][lfidx].lemma_+" id"],table_prefix=schema.table_tokens_lemma_str[tb].split(" | ")[0],pk=True)
                
                elif p in patterns1 and tp_str2.endswith(p):
                    lfidx = index_list[lfidx] 
                else:
                    break
                sq = add_fk_to_table(sq,schema,list_idx[i],lfidx)
    return sq



def add_fk_to_table(sq,schema,sq_idx1,sq_idx2):
    if not get_all_col_from_sq(sq,schema,sq.table_match[sq_idx1][sq_idx2]):
        all_tables = get_all_table_from_sq(sq,schema)
        for tb in all_tables:
            if tb not in sq.table_match[sq_idx1][sq_idx2]:
                for col in schema.tbl_col_idx_back[tb]:
                    if col in schema.foreignKeyDict.keys():
                        for c2 in schema.foreignKeyDict[col]:
                            if schema.column_tokens_table_idx[c2] in sq.table_match[sq_idx1][sq_idx2]:
                                if sq.col_match[sq_idx1][sq_idx2] == []:
                                    sq.col_match[sq_idx1][sq_idx2] = [[col],[1],[1],[col]]
                                elif col not in sq.col_match[sq_idx1][sq_idx2][0]:
                                    sq.col_match[sq_idx1][sq_idx2][0].append(col)
                                    sq.col_match[sq_idx1][sq_idx2][1].append(1)
                                    sq.col_match[sq_idx1][sq_idx2][2].append(1)
                                    sq.col_match[sq_idx1][sq_idx2][3].append(col)
    return sq



def for_each_col(select,token_pattern,question,list_idx,sq,schema):
    if not select:
        return sq
    for i,tps in enumerate(token_pattern):
        if sq.sub_sequence_type[list_idx[i]] == 0 and len(select) == 2 and ((select[0][0] > 0 and select[1][0] == 0) or (select[1][0] > 0 and select[0][0] == 0)):
            if "COL" not in tps and "TABLE" in tps:
                tbm = sq.col_match[list_idx[i]][tps.index("TABLE")]
                if not tbm:
                    tbm = [[],[],[],[]]
                cols = select[0][1] if select[0][0] == 0 else select[1][1]
                tbm[0].extend(cols)
                tbm[1].extend([1]*len(cols))
                tbm[2].extend([1]*len(cols))
                tbm[3].extend(cols)
                sq.col_match[list_idx[i]][tps.index("TABLE")] = tbm

    if (len(select) == 2 and (select[0][0] or select[1][0])) or len(select) == 1:
        correct_select = []
        for sel in select:
            if sel and sel[1]:
                correct_select.extend(sel[1])
        correct_select = set(correct_select)
        for i, col_matchs in enumerate(sq.col_match):
            if sq.sub_sequence_type[i] <= 1:
                for col_match in col_matchs:
                    if col_match:
                        for col in col_match[0]:
                            if col in correct_select:
                                correct_select.remove(col)
        if correct_select:
            for cs in correct_select:
                if "name" in schema.column_tokens_lemma_str[cs] or "title" in schema.column_tokens_lemma_str[cs]:
                    table_id = schema.column_names_original[cs][0]
                    for i, table_matchs in enumerate(sq.table_match):
                        if sq.sub_sequence_type[i] <= 1:
                            for j,table_match in enumerate(table_matchs):
                                for t in table_match:
                                    if t == table_id:
                                        len_match = schema.column_tokens_lemma_str[cs].split(" | ")[0].count(" ")
                                        len_match = 1 if len_match == 0 else 0
                                        if not sq.col_match[i][j]:
                                            sq.col_match[i][j] = [[cs],[1],[len_match],[cs]]
                                        else:
                                            sq.col_match[i][j][0].append(cs)
                                            sq.col_match[i][j][1].append(1)
                                            sq.col_match[i][j][2].append(len_match)
                                            sq.col_match[i][j][3].append(cs)
    return sq


def add_same_col_for_select(sq,schema):
    all_tables = get_all_table_from_sq(sq,schema)
    for i,cols,type_,q_tokens,t_m in zip(range(len(sq.col_match)),sq.col_match,sq.sub_sequence_type,sq.question_tokens,sq.table_match):
        if type_ == 1:
            for z,col in enumerate(cols):
                if col:
                    for d1,d2,d3,d4 in zip(col[0],col[1],col[2],col[3]):
                        already_add = False
                        for ci in schema.same_col_idxs[d1]:
                            if schema.column_tokens_table_idx[ci] in all_tables and ci not in col[0] and schema.column_tokens_table_idx[ci] not in t_m[z]:
                                col[0].append(ci)
                                col[1].append(d2)
                                col[2].append(d3)
                                col[3].append(ci)
                                already_add = True
                        if d1 in schema.foreignKeyDict.keys():
                            for ci in schema.foreignKeyDict[d1]:
                                if schema.column_tokens_table_idx[ci] in all_tables and ci not in col[0] and schema.column_tokens_table_idx[ci] not in t_m[z] and (schema.column_tokens_lemma_str[ci] == schema.column_tokens_lemma_str[d1] or schema.table_tokens_lemma_str[schema.column_tokens_table_idx[ci]].split(" | ")[0] + " " + schema.column_tokens_lemma_str[ci] == schema.column_tokens_lemma_str[d1] or  schema.column_tokens_lemma_str[ci] == schema.table_tokens_lemma_str[schema.column_tokens_table_idx[d1]].split(" | ")[0] + " " + schema.column_tokens_lemma_str[d1] ):
                                    col[0].append(ci)
                                    col[1].append(d2)
                                    col[2].append(d3)
                                    col[3].append(ci)
                                    already_add = True
                        if already_add:
                            break
    return sq


def for_each_analyse(token_pattern,question,list_idx,sq,schema):
    add_to_each = []
    add_to_each_p = []
    each_tables = []
    for i,tps in enumerate(token_pattern):
        tp_str = " ".join(tps)
        real_tp_str = " ".join([ t for t in tps if t not in ["#","?"] ])
        real_tp_list = [ (t,j) for j,t in enumerate(tps) if t not in ["#","?"] ]
        if "IN each ST" in real_tp_str or "IN each ST NN" in real_tp_str or "COL each TABLE" in real_tp_str:
            each_idx = tps.index("each")
            st_idx = each_idx+1
            each_tables = sq.table_match[list_idx[i]][st_idx]
            for tb in sq.table_match[list_idx[i]][st_idx]:
                for k,cc in enumerate(schema.tbl_col_tokens_lemma_str[tb]):
                    if ("first" not in cc and "last" not in cc and cc.endswith(" name")) or cc.endswith(" title") or cc == "name" or cc == "title":
                        add_to_each.append(schema.tbl_col_idx_back[tb][k])
                    if schema.tbl_col_idx_back[tb][k] in schema.primaryKey:
                        add_to_each_p.append(schema.tbl_col_idx_back[tb][k])

    if add_to_each:
        not_add = False
        for i,tps in enumerate(token_pattern):
            for col in sq.col_match[list_idx[i]]:
                if col:
                    for c in col[0]:
                        if c in add_to_each or c in add_to_each_p:
                            not_add = True
        if not not_add:
            for i,tps in enumerate(token_pattern):
                tp_str = " ".join(tps)
                real_tp_str = " ".join([ t for t in tps if t not in ["#","?"] ])
                real_tp_list = [ (t,j) for j,t in enumerate(tps) if t not in ["#","?"] ]
                if "IN each ST" in real_tp_str or "IN each ST NN" in real_tp_str or "COL each TABLE" in real_tp_str:
                    for j,c in enumerate(add_to_each):
                        if j == 0:
                            remove_other_cols(sq,schema,list_idx[i],st_idx)
                        if sq.col_match[list_idx[i]][st_idx]:
                            sq.col_match[list_idx[i]][st_idx][0].append(c)
                            sq.col_match[list_idx[i]][st_idx][1].append(1)
                            sq.col_match[list_idx[i]][st_idx][2].append(1)
                            sq.col_match[list_idx[i]][st_idx][3].append(c)
                        else:
                            sq.col_match[list_idx[i]][st_idx] =  [[c],[1],[1],[c]]
                            


def get_other_tables(sq, schema, sq_i, sq_j):
    all_t = []
    for ts in sq.table_match:
        for t in ts:
            all_t.extend(t)
    for i,cols in enumerate(sq.col_match):
        for j,cs in enumerate(cols):
            if sq_i == i and sq_j == j:
                continue
            if cs:
                for c in cs[0]:
                    all_t.append(schema.column_names_original[c][0])
    return all_t

def remove_other_cols(sq, schema, sq_i, sq_j):
    if sq.col_match[sq_i][sq_j]:
        tbs = get_other_tables(sq, schema, sq_i, sq_j)
        for i in range(len(sq.col_match[sq_i][sq_j][0])-1,-1,-1):
            if schema.column_names_original[sq.col_match[sq_i][sq_j][0][i]][0] not in tbs:
                del sq.col_match[sq_i][sq_j][0][i]
                del sq.col_match[sq_i][sq_j][1][i]
                del sq.col_match[sq_i][sq_j][2][i]
                del sq.col_match[sq_i][sq_j][3][i]
            if not sq.col_match[sq_i][sq_j][0]:
                sq.col_match[sq_i][sq_j] = []




if __name__ == '__main__':
    print("start ")
    args = construct_hyper_param()
    sql_json = json.load(open(os.path.join(args.in_file), 'r'))
    table_json = json.load(open(args.table_file, 'r'))
    _tokenizer = get_spacy_tokenizer()
    _concept_word = pickle.load(open(os.path.join("data/conceptnet.pkl"),'rb'))
    lstm = MyStemmer()

    ONE_ID = []
    # tmp_list = []
    table_dict = {}
    all_schema = {}
    for table in table_json:
        table_dict[table['db_id']]=table
        if not ONE_ID:
            all_schema[table["db_id"]] = Schema_Token(_tokenizer,lstm,table,_concept_word)

    for i,sql in enumerate(sql_json) :
        if ONE_ID and i not in ONE_ID:
            continue
        sql["question_or"] = sql["question"]
        for loop in range(3):
            print(i)
            if sql["db_id"] not in all_schema.keys():
                all_schema[sql["db_id"]] = Schema_Token(_tokenizer,lstm,table_dict[sql["db_id"]],_concept_word)
            
            sq = SubQuestion(sql["question"],sql["question_type"],sql["table_match"],sql["question_tag"],sql["question_dep"],sql["question_entt"],sql,run_special_replace=True)
            select_list,select_table,select_list_idx,select_db_idx = sq.sentence_combine()
            others_list,others_table,others_list_idx,others_db_idx = sq.sentence_combine(combine_type=1)
            qsql = QuestionSQL(sq,_tokenizer)
            sq.tokenize(qsql)

            try:
                select,where,group,order,distinct = select_analyze(lstm, select_list,select_table, select_list_idx, all_schema[sql["db_id"]],sq,qsql)
            except:
                select = []
            if not select:
                select = []

            sq = SubQuestion(sql["question"],sql["question_type"],sql["table_match"],sql["question_tag"],sql["question_dep"],sql["question_entt"],sql,run_special_replace=False)
            select_list,select_table,select_list_idx,select_db_idx = sq.sentence_combine()
            others_list,others_table,others_list_idx,others_db_idx = sq.sentence_combine(combine_type=1)
            qsql = QuestionSQL(sq,_tokenizer)
            sq.tokenize(qsql)

            others,_,select_token_pattern = others_analyze(select_list,select_table, select_list_idx, all_schema[sql["db_id"]],sq,qsql,True,select_db_idx,no_pattern=True)
            others,_,others_token_pattern = others_analyze(others_list,others_table, others_list_idx, all_schema[sql["db_id"]],sq,qsql,in_db_match=others_db_idx)
            for sid in select_list_idx:
                sq.col_match[sid] = []
            others,_,select_token_pattern = others_analyze(select_list,select_table, select_list_idx, all_schema[sql["db_id"]],sq,qsql,True,select_db_idx,no_pattern=True)
            others,_,select_token_pattern = others_analyze(select_list,select_table, select_list_idx, all_schema[sql["db_id"]],sq,qsql,True,select_db_idx,True)

            sq,others_token_pattern = sgrsm_add_col_match(others_token_pattern,others_list,others_list_idx,sq,all_schema[sql["db_id"]])
            sq,select_token_pattern = sgrsm_add_col_match(select_token_pattern,select_list,select_list_idx,sq,all_schema[sql["db_id"]])
            
            sq = select_col_and_col_of(select_token_pattern,select_list,select_list_idx,sq,all_schema[sql["db_id"]],len(select))
            sq = for_each_col(select,select_token_pattern,select_list,select_list_idx,sq,all_schema[sql["db_id"]])
            sq = add_same_col_for_select(sq,all_schema[sql["db_id"]])

            for sl in others_token_pattern:
                for j in range(len(sl)):
                    sl[j] = sl[j].replace("SJJS","JJS")
                    sl[j] = sl[j].replace("SGRSM","GRSM")
            
            for sl in select_token_pattern:
                for j in range(len(sl)):
                    if sl[j]  == "COL":
                        sl[j] = "SC"
                    elif sl[j]  == "TABLE":
                        sl[j] = "ST"
                    elif sl[j]  == "TABLE-COL":
                        sl[j] = "STC"
                    sl[j] = sl[j].replace("SJJS","JJS")
                    sl[j] = sl[j].replace("SGRSM","GRSM")

            return_result = others_where_order_analyse(others_token_pattern,others_list,select_list,select)
            return_result += select_where_order_analyse(select_token_pattern,select_list,select)


            if len(return_result) > 1:
                for rr in range(1,len(return_result)):
                    return_result[0][0] += return_result[rr][0]
                    return_result[0][1] += return_result[rr][1]
                    return_result[0][2] += return_result[rr][2]
                    return_result[0][3] += return_result[rr][3]
            return_result = return_result[0]
                
            if not args.keep_or:
                sq = question_modify(others_token_pattern,others_list,others_list_idx,sq,all_schema[sql["db_id"]])
                sq = question_modify(select_token_pattern,select_list,select_list_idx,sq,all_schema[sql["db_id"]])
            sq = not_col_sub_q(sq,all_schema[sql["db_id"]])
            for_each_analyse(select_token_pattern,select_list,select_list_idx,sq,all_schema[sql["db_id"]])
            for_each_analyse(others_token_pattern,others_list,others_list_idx,sq,all_schema[sql["db_id"]])
            
            print(qsql.question_type)
            oq = copy.deepcopy(sql["question"])
            sql["question"] = sq.gennerate_question()                
            print(sql["question"])
            
            if oq == sql["question"]:
                sql["db_match"] = sq.gennerate_db_match()
                sql["col_match"] = sq.gennerate_col_match(all_schema[sql["db_id"]])
                sql["table_match"] = sq.gennerate_table_match(all_schema[sql["db_id"]],sql["table_match"])
                sql["pattern_tok"] = sq.gennerate_pattern_tok()
                sql["question_lemma"] = sq.gennerate_question_lemma()
            
                print(sql["table_match"])
                print(sql["col_match"])
                print(sql["pattern_tok"])
                print(sql["db_match"])

                print()
                print()
                break
            else:
                sql["db_match"],sql["col_match"] = sq.gennerate_original_matchs()
            
            if "full_db_match" in sql and len(sql["full_db_match"]) != len(sql["db_match"]):
                sql["full_db_match"] = sql["db_match"]

        assert sql["question"].count(" ") + 1 == len(sql["table_match"])
        sql["question_toks"] = sql["question"].split(" ")
    if not ONE_ID:
        json.dump(sql_json,open(args.out_file,'w'), indent=2)

