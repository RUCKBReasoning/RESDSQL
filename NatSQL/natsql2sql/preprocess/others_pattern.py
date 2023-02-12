import os,copy
import re
from .match import COUNTRYS_DICT,S_ADJ_WORD_DIRECTION,AGG_WORDS,AGG_OPS,WHERE_STOP_WORDS as STOP_WORDS,STOP_WORDS as ALL_STOP_WORDS,ABSOLUTELY_GRSM_DICT,RELATED_WORD,NEGATIVE_WORDS,SPECIAL_DB_WORD,INFORMATION_WORDS,ABSOLUTELY_GREATER_DICT,ABSOLUTELY_SMALLER_DICT,ALL_JJS,SELECT_FIRST_WORD,NOT_STAR_WORD,ALL_IMPORTANT_PATTERN_TOKENS
from .utils import str_is_num,number_back,str_is_date,look_for_closest_table_idx,is_there_sgrsm_and_gr_or_sm,get_punctuation_word
from .TokenString import SToken as Token
from .col_match import of_for_structure_in_col
from .db_match import datebase_match_tables,return_result,get_match_col,get_database_string
from .Schema_Token import Schema_Token
from .stemmer import MyStemmer

DISTINCT_WORDS = ["distinct","different","distinctive","unique"]

lstem = MyStemmer()
DEBUG_PATTERN = False

DELETE_WORDS = {"the","is","was","are","were","been", "am","a", "it", "an","its","their","ours","his",  "your", "her", "their", "and", 
 "this","that", "'",'"','of','some','any','many', "such","both","either","every","'re", "'ll", "'s", "'d", "'m", "'ve", "will", "would","could","can","do","did",",",":",".","?","(",")" }




PATTERN_FUN = []
PATTERNS_TOKS = []

def create_pattern_toks(pt, ps):
    for ptts in ps:
        p = []
        for ptt in ptts.split(" | "):
            p.append(ptt.split(" "))
        pt.append(p)
    return pt


def pattern_token_to_new_style(word, tok, all_tokens, schema, table_idxs=[-1], guess=True, col_match=[]):
    if not guess and word not in ["SGRSM","SJJS"] and tok.lemma_ in S_ADJ_WORD_DIRECTION.keys():
        word = "SGRSM"
        if tok.tag_  in ["JJS","RBS"]:
            word = "SJJS"
        
       
    if word in ["SGRSM","SJJS"]:
        final = ""
        find_sgrsm = is_there_sgrsm_and_gr_or_sm(all_tokens,tok,0)
        if find_sgrsm:
            final = find_sgrsm

        if not find_sgrsm:
            key = tok.lemma_ if tok.lemma_ in S_ADJ_WORD_DIRECTION.keys() else lstem.stem(tok.text)
            _,_,sgrsm_word = get_AWD_column(key, table_idxs, schema, restrict=True, all_tokens=all_tokens, re_all_word=True)
            if sgrsm_word:
                if type(sgrsm_word) == str and " " in sgrsm_word:
                    sgrsm_word = sgrsm_word.split(" ")
                if type(sgrsm_word) is list:
                    f_tmp = None
                    for s_adj in S_ADJ_WORD_DIRECTION[key]:
                        if s_adj[0] in sgrsm_word:
                            if s_adj[2] and (f_tmp is None or f_tmp == "SM_"):
                                f_tmp = "SM_"
                            elif s_adj[2] == 0 and (f_tmp is None or f_tmp == "GR_"):
                                f_tmp = "GR_"
                            else:
                                f_tmp = None
                                break
                    if f_tmp:
                        final = f_tmp
                else:
                    for s_adj in S_ADJ_WORD_DIRECTION[key]:
                        if s_adj[0] == sgrsm_word:
                            if s_adj[2]:
                                final = "SM_"
                            else:
                                final = "GR_"
            elif col_match and key in S_ADJ_WORD_DIRECTION.keys():
                there_is_date = False
                desc_asc = 0
                for w in S_ADJ_WORD_DIRECTION[key]:
                    if w[0] in "date":
                        there_is_date = True
                        desc_asc = w[2] 
                        break
                if there_is_date:
                    for cs in col_match:
                        if cs:
                            for c in cs[0]:
                                if schema.column_types[c] in ["year","time"]:
                                    if desc_asc:
                                        final = "SM_"
                                    else:
                                        final = "GR_"
            else:
                final = ""
                word = word[1:]
        word = final + word
        
    if word in ["GRSM","JJS"]:
        if tok.lemma_ in ABSOLUTELY_GREATER_DICT.keys():
            return "GR_" + word
        else:
            return "SM_" + word
    elif word in ["GR","SM"]:
        return word + "_GRSM"
    return word



def word_match(word,tok,table_match,col_match,entt,db_match,all_tokens,idx,schema,all_star=False):
    if word[0] == "$":
        word = word[1:]
    if word == "_":
        return True
    elif word == "*":
        if table_match or col_match or db_match or str_is_num(tok.text) or (idx > 0 and tok.lemma_ in ["and","or"]) or tok.lemma_ in AGG_WORDS or tok.lemma_ in ABSOLUTELY_GREATER_DICT.keys() or tok.lemma_ in S_ADJ_WORD_DIRECTION.keys() or entt in SPECIAL_DB_WORD or get_punctuation_word(all_tokens,idx) or tok.lemma_ in NOT_STAR_WORD or tok.tag_ in ["JJS","JJR","RBR","RBS"] or (tok.lower_ in COUNTRYS_DICT and not tok.text.islower()):
            return False
        elif all_star and (tok.text in ALL_IMPORTANT_PATTERN_TOKENS or tok.lemma_ in ALL_IMPORTANT_PATTERN_TOKENS ):
            return False
        return True
    elif word == "?":#STOP WORD:
        if tok.lemma_ in STOP_WORDS or tok.text in STOP_WORDS:
            return True
    elif word.isupper():
        if word in ["TABLE","_TABLE_","PTABLE"]: # _TABLE_ is for only one table token. TABLE is for at least one token.
            if table_match and (word != "PTABLE" or not col_match):
                return True
        elif word in ["COL","TCOL","PCOL","BCOL"]:
            if col_match:
                if tok.tag_ in ["JJS","RBS"] and idx + 1 < len(all_tokens) and tok.lemma_ == all_tokens[idx + 1].lemma_:
                    return False
                if word == "COL":
                    return True
                if (word == "PCOL" and not table_match) or word in ["BCOL","TCOL"]:
                    bcol = False
                    for ccc in col_match[0]:
                        if schema.column_types[ccc] == "boolean":
                            bcol = True
                            break
                    if (bcol and word == "BCOL") or (not bcol and word in ["PCOL","TCOL"]):
                        return True
        elif word == "NOT":
            if tok.text in NEGATIVE_WORDS:
                return True
        elif word == "DB":
            if db_match and not (tok.tag_ == "IN" and idx == 0) :
                return True
        elif word in ["PDB","PDB_C"]:
            if word == "PDB" and (table_match or col_match or db_match):
                return False
            if get_punctuation_word(all_tokens,idx):
                return True
        elif word == "SDB":
            if table_match or col_match or db_match:
                return False
            if entt in SPECIAL_DB_WORD:
                return True
        elif word in ["SGRSM","SJJS"]:
            if (word == "SJJS" and tok.tag_ not in ["JJS","RBS"]) or (word != "SJJS" and tok.tag_ in ["JJS","RBS"]):
                return False
            if tok.lemma_ in S_ADJ_WORD_DIRECTION.keys() or lstem.stem(tok.text) in S_ADJ_WORD_DIRECTION.keys():
                return True
        elif word in ["GRSM","JJS","GR","SM"]:
            text_stem = lstem.stem(tok.text)
            if (word == "JJS" and tok.tag_ not in ["JJS","RBS"]) or (word != "JJS" and tok.tag_ in ["JJS","RBS"]):
                return False
            elif word == "GR":
                if tok.lemma_ in ABSOLUTELY_GREATER_DICT.keys() or text_stem in ABSOLUTELY_GREATER_DICT.keys():
                    return True
            elif word == "SM":
                if tok.lemma_ in ABSOLUTELY_SMALLER_DICT.keys() or text_stem in ABSOLUTELY_SMALLER_DICT.keys():
                    return True
            else:
                if tok.lemma_ in ABSOLUTELY_GRSM_DICT.keys() or text_stem in ABSOLUTELY_GRSM_DICT.keys():
                    return True
        elif word == "NUM":
            return str_is_num(tok.text)
        elif word == "AGG":
            if tok.lemma_ in AGG_WORDS:
                return True
        elif word == "DATE": 
            # if tok.tag_ in ["DATE","CARDINAL"]:
            #     return True
            if str_is_date(tok.text,all_tokens,idx):
                return True
        elif word == "YEAR" and str_is_date(tok.text,all_tokens,idx) == word: 
            return True
        elif word == "UPPER":
            if tok.text.isupper():
                return True
        elif word == entt:
            if table_match or col_match:
                return False
            return True
        elif word == "WP" and tok.text in {"where", "because", "who", "while", "whom", "then", "that", "when",  "which", "whose" }:
            return True
        elif word == tok.tag_ or word == tok.tag_[0]:
            if table_match or col_match or db_match or str_is_num(tok.text) or (idx > 0 and tok.lemma_ in ["and","or"]) or tok.lemma_ in AGG_WORDS or tok.lemma_ in ABSOLUTELY_GREATER_DICT.keys() or tok.lemma_ in S_ADJ_WORD_DIRECTION.keys() or entt in SPECIAL_DB_WORD or get_punctuation_word(all_tokens,idx) or tok.lemma_ in NOT_STAR_WORD or tok.tag_ in ["JJS","JJR","RBR","RBS"] or (tok.lower_ in COUNTRYS_DICT and not tok.text.islower()):
                return False
            return True
    else:
        if word == tok.text or word == tok.lemma_ or lstem.stem(tok.text) == word or (tok.lemma_.endswith("ly") and tok.lemma_[:-2] == word):
            return True
    return False


def global_match(pattern,tokens,table_match,col_match,entt,db_match,schema):
    p_num,p_word = pattern.split("&")
    p_num = int(p_num)
    i = 0
    while i < len(tokens):
        if word_match(p_word,tokens[i],table_match[i],col_match[i],entt[i],db_match[i],tokens,i,schema):  
            if p_num == 1:
                return True,i
            p_num -= 1
            if p_word in ["TABLE","COL","DB","PDB","PDB_C","SDB","$TABLE","$COL","$DB","$PDB","$PDB_C","$SDB"]:
                    while True:
                        i += 1
                        if i >= len(table_match) or (p_word in ["DB","$DB"] and db_match[i] != db_match[i-1]) or not word_match(p_word,tokens[i],table_match[i],col_match[i],entt[i],db_match[i],tokens,i,schema):
                            i -= 1
                            break
        i += 1
    return False,-1

def pattern_match(lss, pattern, pattern_idx, sentence_ts, table_match, col_match, entt, db_match, schema, return_special_idx = -1, pattern_fun=PATTERN_FUN):
    
    def can_delete(i,sentence_ts):
        if sentence_ts.tokens[i].text not in DELETE_WORDS:
            return False
        if i > 0 and i < len(sentence_ts.tokens)-1 and sentence_ts.tokens[i-1].text == "'" and sentence_ts.tokens[i+1].text == "'":
            return False
        return True

    def if_table_col_continue_not_same(pw,i,table_match,col_match):
        if pw in ["TABLE","$TABLE"]:
            if table_match[i] == table_match[i-1]:
                return False
            for t in table_match[i]:
                if t in table_match[i-1]:
                    return False
            return True
        elif pw in ["COL","PCOL","TCOL","BCOL","$COL","$PCOL","$TCOL","$BCOL"]:
            if col_match[i] == col_match[i-1]:
                return False
            elif col_match[i][0] == col_match[i-1][0]:
                return False
            for c in col_match[i][0]:
                if c in col_match[i-1][0]:
                    return False
            return True
        return False

    """
    lss:
        useless now
    pattern_idx:
        return function index. also is the pattern index. when it is -1, return the first function.
    return_special_idx:
        return the token idx responding to the pattern token idx.(input pattern token idx)
    """
    i = 0
    negative = False
    pattern_for_ml = []
    if pattern[0] == "@":
        for j, pw in enumerate(pattern):
            if pw in ["@"]:
                continue
            match,pw_idx = global_match(pw,sentence_ts.tokens,table_match,col_match,entt,db_match,schema)
            if not match:
                return None,False,None
            if return_special_idx == j:
                return pw_idx
    else:
        all_star_p = True if pattern.count("*") == len(pattern) else False
        for j, pw in enumerate(pattern):
            miss_match = True
            if pw in ["+"]:
                continue
            if i >= len(table_match):
                return None,False,None

            if pw == sentence_ts.tokens[i].lower_ or pw == sentence_ts.tokens[i].lemma_:
                if return_special_idx == j:
                    return i
                i += 1
                pattern_for_ml.append(pw)
                continue
            
            if pw[0] == "-":
                if pw[1:] == sentence_ts.tokens[i].lower_ or pw[1:] == sentence_ts.tokens[i].lemma_:
                    return None,False,None
                continue

            while can_delete(i,sentence_ts) or (sentence_ts.tokens[i].tag_ == "DT" and  sentence_ts.tokens[i].text not in ["each"] and sentence_ts.tokens[i].text not in NEGATIVE_WORDS and sentence_ts.tokens[i].text not in DELETE_WORDS): # delete STOP WORD.
                i += 1
                if i == len(table_match):
                    i -= 1
                    break # break the while
                if pw == sentence_ts.tokens[i].lower_ or pw == sentence_ts.tokens[i].lemma_:
                    break
                pattern_for_ml.append("#")

            if pw == "NO#" and i != 0:
                return None,False,None
            elif pw == "NO#" and i == 0:
                continue

            if pw == sentence_ts.tokens[i].lower_ or pw == sentence_ts.tokens[i].lemma_:
                if return_special_idx == j:
                    return i
                i += 1
                pattern_for_ml.append(pw)
                continue
                
            if sentence_ts.tokens[i].text in NEGATIVE_WORDS:
                i += 1
                if pw == "NOT":
                    pattern_for_ml.append("NOT")
                    continue
                elif pattern[0] == "+":
                    return None,False,None
                negative = True
                if i == len(table_match): # work for last while and if
                    i -= 1
                else:
                    pattern_for_ml.append("NOT")

            while pw in ["COL","$COL"] and table_match[i] and (not col_match[i] or (i + 1 < len(col_match) and col_match[i] != col_match[i+1])): # delete the table match before col
                i += 1
                pw = "COL" # when it delete the table match, here must be COL not $COL
                if i == len(table_match):
                    i -= 1
                    break # break the while
                pattern_for_ml.append("TABLE")

            if not word_match(pw,sentence_ts.tokens[i],table_match[i],col_match[i],entt[i],db_match[i],sentence_ts.tokens,i,schema,all_star_p):
                if pw[0] != "$":
                    return None,False,None
                i -= 1
                miss_match = False
            else:
                if pw in ["TABLE","COL","PCOL","TCOL","BCOL","DB","PDB","PDB_C","SDB","$TABLE","$COL","$PCOL","$TCOL","$BCOL","$DB","$PDB","$PDB_C","$SDB"]:
                    while True:
                        i += 1
                        if i >= len(table_match) or (pw in ["DB","$DB"] and db_match[i] != db_match[i-1]) or not word_match(pw,sentence_ts.tokens[i],table_match[i],col_match[i],entt[i],db_match[i],sentence_ts.tokens,i,schema):
                            i -= 1
                            break # break the while
                        elif if_table_col_continue_not_same(pw,i,table_match,col_match):
                            pw in ["TABLE","COL","PCOL","TCOL","BCOL","$TABLE","$COL","$PCOL","$TCOL","$BCOL"]
                            i -= 1
                            break # break the while
                        if pw[0] != "$":
                            pattern_for_ml.append(pw)
                        else:
                            pattern_for_ml.append(pw[1:])

            if return_special_idx == j:
                return i
            if return_special_idx == -1:
                if sentence_ts.tokens[i].text in NEGATIVE_WORDS:
                    pattern_for_ml.append("NOT")
                elif pw[0] != "$":
                    pattern_for_ml.append(pattern_token_to_new_style(pw,sentence_ts.tokens[i],sentence_ts.tokens,schema,guess=False))
                elif miss_match:
                    pattern_for_ml.append(pattern_token_to_new_style(pw[1:],sentence_ts.tokens[i],sentence_ts.tokens,schema,guess=False))
            i += 1
        if i != len(table_match) and pattern[0] != '@': # if the rest words are not STOP WORDS, return None
            for j in range(i,len(table_match),1):
                if sentence_ts.tokens[j].text not in STOP_WORDS and sentence_ts.tokens[j].text not in DELETE_WORDS:
                    return None,False,None
                else:
                    pattern_for_ml.append("#")
    if pattern_idx < 0:
        return pattern_fun[0],negative,pattern_for_ml
    return pattern_fun[pattern_idx],negative,pattern_for_ml


def len_sentence(sentence_ts):
    if sentence_ts.tokens[-1].text in ['.',',','?','!']:
        return len(sentence_ts.tokens) - 1
    return len(sentence_ts.tokens)


def pattern_reconize(sentence_ts,table_match,col_match,entt,schema,table_idxs,pattern_toks = PATTERNS_TOKS,pattern_fun=PATTERN_FUN,in_db_match=None,db_table_idxs=None):
    lss = len_sentence(sentence_ts)
    db_match = [ [] for i in range(len(sentence_ts.tokens))]
    if in_db_match:
        db_match = in_db_match
    for pidx, patterns in enumerate(pattern_toks):
        for p in patterns:
            if p == ['START', 'SEARCH', 'DATABASE'] and not in_db_match: # create_db_match
                start_idx = 0
                not_see_punt = True
                while True:
                    if start_idx >= lss:
                        break
                    if sentence_ts.tokens[start_idx].text in ["'","?",",","."]:
                        if sentence_ts.tokens[start_idx].text in ["'",'"']:
                            not_see_punt = False
                        start_idx += 1
                        continue
                    elif (not_see_punt or not get_punctuation_word(sentence_ts.tokens,start_idx)) and not sentence_ts.tokens[start_idx].text.isupper() and \
                        (sentence_ts.tokens[start_idx].lower_ in ALL_STOP_WORDS or len(sentence_ts.tokens[start_idx].text)<=3 or sentence_ts.tokens[start_idx].lower_ in SELECT_FIRST_WORD or sentence_ts.tokens[start_idx].text in ABSOLUTELY_GRSM_DICT.keys()) and\
                             (sentence_ts.tokens[start_idx].text.islower() or start_idx==0 or (not sentence_ts.tokens[start_idx].text.isalpha() and len(sentence_ts.tokens[start_idx].text)==1) ) and\
                                  (sentence_ts.tokens[start_idx].tag_ in ["IN","DT","CC","RP","JJR","JJS","VBP","VBD","VBD","VBZ","VB"] or sentence_ts.tokens[start_idx].lemma_ in ["be","have","me","do","-PRON-","top"] or  (not sentence_ts.tokens[start_idx].text.isalpha() and len(sentence_ts.tokens[start_idx].text)==1) ):#(start_idx>1 and sentence_ts.tokens[start_idx-1].lemma_ in ["of"]) or
                        start_idx += 1
                        continue

                    if db_table_idxs:
                        res = datebase_match_tables(schema,sentence_ts.tokens[start_idx],start_idx,sentence_ts.tokens,db_table_idxs,True,True)
                    else:
                        res = datebase_match_tables(schema,sentence_ts.tokens[start_idx],start_idx,sentence_ts.tokens,table_idxs,True,True)
                    
                    if res and (not_see_punt or not get_punctuation_word(sentence_ts.tokens,start_idx)) and sentence_ts.tokens[start_idx].text.islower() and (table_match[start_idx] or col_match[start_idx]):
                        new_res = []
                        new_res_str = []
                        for rss in res:
                            for rs in rss:
                                if str(rs) not in new_res_str:
                                    new_res.append([rs])
                                    new_res_str.append(str(rs))
                        res = new_res
                        res_is_ok = False
                        for rs in res:
                            for r in rs:
                                if r[1][1] - r[1][0] >= 1:
                                    res_is_ok = True
                                    for rr in range(r[1][0],r[1][1]+1,1):
                                        table_match[rr] = []
                                        col_match[rr] = []
                                
                        if not res_is_ok:
                            res = None
                    
                    if not res:
                        start_idx += 1
                        continue
                    else:
                        for rs in res:
                            for r in rs:
                                for rr in range(r[1][0],r[1][1]+1,1):
                                    db_match[rr].append(r)
                        start_idx = r[1][1]+1
                for i in range(len(db_match)):
                    db_match[i] = return_result(db_match[i])
            else:
                p_fun,negative,p_all_toks = pattern_match(lss, p, pidx, sentence_ts,table_match,col_match,entt,db_match,schema,-1,pattern_fun)
                if p_fun:
                    if DEBUG_PATTERN:
                        print("MAIN PATTERN MATCH:")
                        print(p)
                        print(sentence_ts)
                        print()
                    return p_fun(p, sentence_ts,table_match,col_match,entt,schema,db_match,table_idxs,negative),db_match
    return (None,None,None,None,None,None),db_match




def find_table_idx(col_id,col_table,idx_start,schema):
    for i,t in enumerate(col_table):
        if i > idx_start and t:
            return t
    if col_id:
        table_idxs = []
        for col in col_id:
            table_idxs.append(schema.column_tokens_table_idx[col])
        return table_idxs
    else:
        return [-1]


def get_AWD_column(word, table_idxs, schema, restrict=False, all_tokens=None, re_all_word = False):
    if word not in S_ADJ_WORD_DIRECTION.keys():
        word = schema.lemmanize(word)
        if word not in S_ADJ_WORD_DIRECTION.keys():
            return -1, -2, None
    
    all_words = []
    if all_tokens:
        if type(all_tokens[0]) == list:
            all_words = [tok[0].text for tok in all_tokens]
            all_words.extend([tok[0].lemma_ for tok in all_tokens])
            all_words = set(all_words)
            for sgrsm_word in S_ADJ_WORD_DIRECTION[word]:
                for tok in all_tokens:
                    if tok[1] > 0 and (tok[0].text == sgrsm_word[0] or tok[0].lemma_ == sgrsm_word[0]):
                        if (tok[0].text in schema.column_tokens_lemma_str or tok[0].lemma_ in schema.column_tokens_lemma_str):
                            return 0,[-1],sgrsm_word[0]
        else:
            all_words = [tok.text for tok in all_tokens]
            all_words.extend([tok.lemma_ for tok in all_tokens])
            all_words = set(all_words)
                    

    col_id_list = []
    r_word_list = []
    r_word_set = set()

    if word in ["long","short"] and "song" in all_words:
        word = word + "song"

    for sgrsm_word in S_ADJ_WORD_DIRECTION[word]:
        agg_id, col_id = one_word_to_column_match(table_idxs,sgrsm_word[0],cross_table_search = False,schema = schema,use_concept_match=False, allow_list = True)
        
        if restrict and col_id:
            if not r_word_set or len(col_id) == 1:
                col_id_list.extend(col_id)
                r_word_list.append(sgrsm_word[0])
                r_word_set.add(sgrsm_word[0])
        elif not restrict and col_id and len(col_id)==1 and agg_id >= 0:
            return agg_id, col_id[0], sgrsm_word

    if restrict and len(col_id_list)>1 and len(r_word_list)>1:
        r_word_set = set()
        for col in col_id_list:
            for sgrsm_word in S_ADJ_WORD_DIRECTION[word]:
                if sgrsm_word[0] == schema.column_tokens_lemma_str[col] or schema.column_tokens_lemma_str[col].endswith(sgrsm_word[0]):
                    r_word_set.add(sgrsm_word[0])
    if restrict and len(set(col_id_list))==1:
        return 0, [col_id_list[0]], schema.column_tokens_lemma_str[col_id_list[0]]
    elif restrict and col_id_list and len(r_word_set)==1:
        return 0, col_id_list, list(r_word_set)[0]
    if restrict and r_word_set and col_id_list:
        if word in ["old","young"] and "age" in r_word_set:
            for i in range(len(col_id_list)-1,-1,-1):
                if "age" not in schema.column_tokens_text_str[col_id_list[i]]:
                    del col_id_list[i]
            return 0, col_id_list, "age"
        return -1, [-2], None
    if re_all_word and restrict and col_id_list and r_word_list:
        return 0, col_id_list, r_word_list

    if table_idxs != [-1]:
        for sgrsm_word in S_ADJ_WORD_DIRECTION[word]:
            agg_id, col_id = one_word_to_column_match(table_idxs,sgrsm_word[0],cross_table_search = True,schema = schema,use_concept_match=False)
            if restrict and col_id:
                col_id_list.extend(col_id)
                r_word_set.add(sgrsm_word[0])
            elif not restrict and col_id and len(col_id)==1 and agg_id >= 0:
                return agg_id, col_id[0], sgrsm_word
    if restrict and col_id_list and len(r_word_set)==1:
        return 0, col_id_list, list(r_word_set)[0]

    if restrict:
        return -1, [-2], None
    return -1, -2, None


def get_col_from_related_word(word, table_idxs, schema, restrict=False):
    if word in RELATED_WORD.keys():
        col_id_list = []
        r_word_list = set()
        for r_word in RELATED_WORD[word]: # do not cross table:
            agg_id, col_id = one_word_to_column_match(table_idxs,r_word,cross_table_search = False,schema = schema,use_concept_match=False)
            if restrict and col_id:
                col_id_list.extend(col_id)
                r_word_list.add(r_word)
            elif not restrict and col_id and len(col_id)==1 and agg_id >= 0:
                return agg_id, col_id[0], r_word

        if restrict and col_id_list and len(r_word_list)==1:
            return 0, col_id_list, list(r_word_list)[0]
        if restrict and r_word_list and col_id_list:
            return -1, [-2], None

        for r_word in RELATED_WORD[word]: # cross table:
            agg_id, col_id = one_word_to_column_match(table_idxs,r_word,cross_table_search = True,schema = schema,use_concept_match=False)
            if restrict and col_id:
                col_id_list.extend(col_id)
                r_word_list.add(r_word)
            elif not restrict and col_id and len(col_id)==1 and agg_id >= 0:
                return agg_id, col_id[0], r_word
        if restrict and col_id_list and len(r_word_list)==1:
            return 0, col_id_list, list(r_word_list)[0]
    if restrict:
        return -1, [-2], None
    return -1, -2, None




def one_word_to_column_match(table_idx,select_words,cross_table_search,schema:Schema_Token,use_concept_match=True,only_two_match_fuc=False, allow_list = True):
    agg_id, col_id = schema.one_word_to_tables_column_match(table_idx, select_words,table_idx[0] >= 0,cross_table_search = cross_table_search,use_concept_match=use_concept_match,only_two_match_fuc=only_two_match_fuc, allow_list = allow_list)
    return agg_id, col_id



def words_to_column_match(table_idx,select_words,sel_word_idxs,available_idxs,cross_table_search,schema:Schema_Token,use_concept_match=True):
    def reorder_col_availabe_word(col_availabe,col_availabe_idx,a_col_idx,jjj):
        word = ""
        for aci,iii in zip(reversed(a_col_idx),range(len(a_col_idx))):
            if iii < jjj:
                continue
            for colw,cola in zip(col_availabe,col_availabe_idx):
                if cola in aci:
                    word += colw
                    word += " "
        word = word.strip()
        return word

    start = 0
    select_word_all = reorder_col_availabe_word(select_words,sel_word_idxs,available_idxs,start)
    select_word = select_word_all
    final_round = False
    if select_word_all.count(" ") == 0:
        agg_id, col_id = schema.one_word_to_tables_column_match(table_idx, select_word,table_idx[0] >= 0,cross_table_search = cross_table_search,use_concept_match=use_concept_match)
    else:
        while True:
            if select_word not in AGG_WORDS:
                agg_id, col_id = schema.one_word_to_tables_column_match(table_idx, select_word,table_idx[0] >= 0,cross_table_search = cross_table_search,use_concept_match=use_concept_match,final_round=final_round)
            if col_id:
                return agg_id, col_id

            if final_round and select_word == select_word_all:
                break
            
            if select_word.count(" ") == 0:
                final_round = True
            else:# remove the first word
                select_word = " ".join(select_word.split(" ")[1:])

            if final_round:
                select_word = select_word_all
            
    return agg_id, col_id

def one_col_match(col,a_col,a_col_idx,o_col_idxs,col_table,schema,table_idx,availble_sent,c_idx,original_sent,q_sql):
    """
    Normally, this function only contain one column information.
    """

    distinct = False
    agg_id, col_id, where, group, order = (-1,[],None,None,None)
    col_inner_idx = []
    idx = 0
    for ac in a_col:
        for c in ac:
            col_inner_idx.append(idx)
        idx += 1
    
    col_availabe = []
    col_availabe_idx = []
    where_words = []
    cross_table_search = True
    meet_each = False
    if len(col_table) == 2 and col_table[0] and col_table[1] and col_table[0] != col_table[1]:
        return agg_id, col_id, table_idx, distinct, where, group, order
    for word,t,o_col_idx,ii in zip(col,col_table,o_col_idxs,range(len(col))):
        if col_inner_idx[-1] > 1 and where_words and not t and col_inner_idx[where_words[-1][1]] == col_inner_idx[ii]:
            where_words.append([word,ii])
        elif not meet_each and t:
            if len(t) >= 1 and t != table_idx: # Update the table_idx
                if of_for_structure_in_col(a_col_idx,q_sql.question_tokens): # only take the first table
                    continue_ = False
                    for iii in range(0,ii,1):
                        if col_table[iii]:
                            continue_ = True
                            break
                    if continue_:
                        continue
                table_idx = t
            
        elif not meet_each and schema.column_contain_word(word) or word in AGG_WORDS or word in INFORMATION_WORDS:
            if ii > 0 and schema.column_contain_word(word) and word in AGG_WORDS and col_table[ii-1]: 
                col_availabe.append(col[ii-1])
                col_availabe_idx.append(o_col_idxs[ii-1])
            col_availabe.append(word)
            col_availabe_idx.append(o_col_idx)
        else:
            if word == "each":
                meet_each = True
            elif word in DISTINCT_WORDS:
                distinct = True
                continue
            where_words.append([word,ii])
    
    if not col_availabe and len(availble_sent) >= 4:
        none_select = True
        for w in where_words:
            if w[0] not in SELECT_FIRST_WORD and w[0] not in ["results","result"]:
                none_select = False
        if none_select:
            return -3, [], table_idx, distinct, where, group, order

    if len(col_availabe) > 3 and col_availabe[0] == "number":
        col_availabe = col_availabe[:3]

    if table_idx and table_idx[0] >= 0 and meet_each and not col_availabe:
        col_id = []
        for ti in table_idx:
            if "name" in schema.tbl_col_tokens_lemma_str[ti]:
                col_id.append(schema.tbl_col_idx_back[ti][schema.tbl_col_tokens_lemma_str[ti].index("name")])
            else:
                pk = schema.primary_keys(ti)
                if pk > 0:
                    col_id.append(pk)
        if col_id:
            return 0, col_id, table_idx, distinct, where, group, order


    if table_idx and table_idx[0] >= 0 and not col_availabe:
        if len(col)==1:
            for t in table_idx:
                if col[0] in schema.tbl_col_tokens_lemma_str[t] or col[0] in schema.tbl_col_tokens_text_str[t] or schema.stem(col[0]) in schema.tbl_col_tokens_stem_str[t]:
                    col_availabe.append(col[0])
        if not col_availabe:
            for tidx in table_idx:
                if "name" in schema.table_col_lemma[tidx]:
                    col_availabe.append("#name")
                elif "title" in schema.table_col_lemma[tidx]:
                    col_availabe.append("#title")
        cross_table_search = False
    
    if where_words and not col_availabe:
        col_availabe = [w[0] for w in where_words]
        where_words = []

    while(col_availabe):
        select_words = " ".join(col_availabe)
        if col_availabe_idx:
            agg_id, col_id = words_to_column_match(table_idx,col_availabe,col_availabe_idx,a_col_idx,cross_table_search,schema)
        else:
            agg_id, col_id = one_word_to_column_match(table_idx,select_words,cross_table_search,schema)


        if agg_id == 0 and select_words in AGG_WORDS: 
            if ii < len(original_sent)-1 and len(original_sent[ii+1])>=1 and original_sent[ii+1][0] in AGG_WORDS:
                agg_id, col_id = (AGG_OPS[AGG_WORDS.index(select_words)],[])
        elif agg_id == 3 and col_id and col_id[0] >= 0 and schema.column_types[col_id[0]] == "number" and col_id[0] not in schema.primaryKey and col_id[0] not in schema.primaryKey and col_id[0] not in schema.foreignKey:
            agg_id = 4
        if (col_id != [] and agg_id >= 0) or len(col_availabe) == 1:
            break
        else: 
            col_availabe = col_availabe[:-1]
    return agg_id, col_id, table_idx, distinct, where, group, order



def pattern_word_guess(tok,table_match,col_match,all_table_match,all_col_match,entt,db_match,all_tokens,schema,table_idxs,idx,show_every_word=False):
    final = ""
    
    if tok.lemma_ in AGG_WORDS:
        if tok.lemma_ == "number" and (idx + 1 >= len(all_tokens) or all_tokens[idx+1].text != "of"):
            pass
        elif not col_match or (1 not in col_match[2] and 2 not in col_match[1]):
            final += ",AGG"

    if db_match:
        final += ",DB"
    if col_match and schema.column_types[col_match[0][0]] == "boolean":
        final = ",BCOL"
    if table_match and col_match:
        if idx > 0 and all_tokens[idx-1].text == "each" and final == ",BCOL":
            final = ",TABLE-COL"
        else:
            final += ",TABLE-COL"
    elif table_match:
        final += ",TABLE"
    elif col_match:
        final += ",COL"
    
    if tok.lemma_ in AGG_WORDS and "AGG" not in final:
        if tok.lemma_ == "number" and (idx + 1 >= len(all_tokens) or all_tokens[idx+1].text != "of"):
            pass
        else:
            final += ",AGG"

    
    if get_punctuation_word(all_tokens,idx):
        final += ",PDB"

    text_stem = lstem.stem(tok.text)
    
    if (tok.lemma_ in S_ADJ_WORD_DIRECTION.keys() or text_stem in S_ADJ_WORD_DIRECTION.keys()) and not (((idx==len(all_tokens)-2 and all_tokens[-1].text in DELETE_WORDS and idx>0 and all_tokens[idx-1].text != "to") or idx==len(all_tokens)-1) and tok.tag_ not in ["JJS","JJR","RBR","RBS"]):
        tmp = None
        if (tok.tag_ in ["JJS","RBS"] and not tok.text.endswith("er")) or ( ( (idx > 0 and all_tokens[idx-1].text in ["most","least"]) or (idx > 1 and all_tokens[idx-2].text in ["most","least",'top'] and str_is_num(all_tokens[idx-1].text)) ) and tok.text in S_ADJ_WORD_DIRECTION.keys()):
            tmp = pattern_token_to_new_style("SJJS", tok, all_tokens, schema,table_idxs,col_match=all_col_match)
            if tmp == "SJJS" and idx + 1 < len(all_table_match) and all_table_match[idx + 1]:
                tmp = pattern_token_to_new_style("SJJS", tok, all_tokens, schema,all_table_match[idx + 1],col_match=all_col_match)
            if idx > 0 and all_tokens[idx-1].text == "least" and tmp != "SJJS":
                tmp = tmp.replace("GR_","SM_") if tmp.startswith("GR_") else tmp.replace("SM_","GR_")
        else:
            tmp = pattern_token_to_new_style("SGRSM", tok, all_tokens, schema,table_idxs,col_match=all_col_match)
        if tmp:
            final += "," + tmp
    
    if (tok.lemma_ in ABSOLUTELY_GRSM_DICT or text_stem in ABSOLUTELY_GRSM_DICT or tok.lemma_ in ALL_JJS) and not (((idx==len(all_tokens)-2 and all_tokens[-1].text in DELETE_WORDS and idx>0 and all_tokens[idx-1].text != "to") or idx==len(all_tokens)-1) and tok.tag_ not in ["JJS","JJR","RBR","RBS"]):
        if tok.lemma_ in ALL_JJS:
            if ALL_JJS[tok.lemma_] in ABSOLUTELY_GREATER_DICT.keys():
                final += ",GR_JJS"
            else:
                final += ",SM_JJS"
        elif tok.text in ["more","most","less","least"] and idx < len(all_tokens)-1 and (all_tokens[idx+1].text in ABSOLUTELY_GRSM_DICT or all_tokens[idx+1].text in S_ADJ_WORD_DIRECTION.keys()):
            pass
        else:
            final += ","
            if tok.lemma_ in ABSOLUTELY_GREATER_DICT.keys() or text_stem in ABSOLUTELY_GREATER_DICT.keys():
                if idx > 0 and all_tokens[idx-1].text == "least":
                    final += "SM_"
                else:
                    final += "GR_"
            else:
                if idx > 0 and all_tokens[idx-1].text == "least":
                    final += "GR_"
                else:
                    final += "SM_"
            if (tok.tag_ in ["JJS","RBS"] and not tok.text.endswith("er")) or tok.text in ["most","least"] or (idx > 0 and all_tokens[idx-1].text in ["most","least"] and tok.text in ABSOLUTELY_GRSM_DICT):
                final += "JJS"
            else:
                final += "GRSM"

    data_or_year = str_is_date(tok.text,all_tokens,idx)
    if data_or_year:
        final += "," + data_or_year
    if str_is_num(tok.text):
        final += ",NUM"   
    if entt in SPECIAL_DB_WORD:
        final += ",SDB"
    if idx > 0 and not tok.text.islower() and tok.text.isalpha():
        final += ",UDB"
    elif tok.lemma_ in ALL_IMPORTANT_PATTERN_TOKENS:
        final += ","+tok.lemma_
    elif tok.text in ALL_IMPORTANT_PATTERN_TOKENS:
        final += ","+tok.text
    elif text_stem in ALL_IMPORTANT_PATTERN_TOKENS:
        final += ","+text_stem
    elif tok.lemma_.endswith("ly") and tok.lemma_[:-2] in ALL_IMPORTANT_PATTERN_TOKENS:
        final += ","+tok.lemma_[:-2]
    elif entt and not tok.tag_.startswith("W") and entt != "DATE":
        final += ","+entt
    if tok.text in NEGATIVE_WORDS or tok.lemma_ in NEGATIVE_WORDS:
        final += ",NOT"

    if ",PDB" in final and ",UDB" in final and ",DB," not in final:
        final = ",PDB"
    elif ",DB," in  final and (",UDB" in final or ",PDB" in final):
        final = ",DB"
    elif ",SDB" in  final and ",UDB" in  final and ",PDB" not in final and ",DB" not in final:
        final = ",SDB"
    pattern_token = final
    if not final:
        final += ",[TAG:"+tok.tag_+",WORD:"+tok.text+"("+tok.lemma_+")"+"]"
        pattern_token += ","+tok.tag_+","+tok.text+","+tok.lemma_

    elif show_every_word:
        final += ",("+tok.lemma_+")"
        pattern_token += ","+tok.lemma_

    return final[1:],pattern_token[1:]



def choose_possible_pattern_token(pattern_str,tokens,idx,raw_pattern_tokens,table_match,col_match,q_tokens,schema):
    def next_pattern_tok(raw_pattern_tokens,col_match,idx,q_tokens,skip_=False):
        for i in range(idx+1,len(raw_pattern_tokens)):
            if raw_pattern_tokens[i][0] not in ["#","and,(and)"]:
                return raw_pattern_tokens[i][0],col_match[i]
            elif skip_ and idx + 1 < len(raw_pattern_tokens):
                return next_pattern_tok(raw_pattern_tokens,col_match,idx+1,q_tokens,skip_)
            elif q_tokens[i].text not in ["the","a","it", "an","its","their","ours","his",  "your", "her", "their", "this","that", 'of','any','many', "such","both","either","every",]:
                return None,None
        return None,None
    final_pattern = ""
    words = pattern_str.split(",")
    for tok in words:
        if tok[0] == "$":
            tok = tok[1:]
        if (tok in STOP_WORDS or tok in {"WP","WP$"}) and tok not in ['and',"but",'or']:
            if tokens[idx].lower_ in {"where", "because", "who", "while", "whom", "then", "that", "when",  "which", "whose" } and idx == 0:
              return "WP"
            return "?"
        elif tok in ALL_IMPORTANT_PATTERN_TOKENS:
            if tok == "COL" and len(words) > 2 :
                tok2,col2 = next_pattern_tok(raw_pattern_tokens,col_match,idx,q_tokens)
                tok3,col3 = next_pattern_tok(raw_pattern_tokens,col_match,idx,q_tokens,True)
                if tok2 and tok2.startswith("COL"):
                    for cc2 in col2[0]:
                        if cc2 in col_match[idx][0]:
                            return tok
                    col_match[idx] = []
                    continue
                elif tok2 and (tok2.startswith("TABLE,(") or tok2.startswith("ST,(")) and ("GR_SJJS" in words or "SM_SJJS" in words):
                    col_match[idx] = []
                    continue
                elif tok3 and "AGG" in tok3 and "AGG" in words:
                    col_match[idx] = []
                    continue
                elif tok2 and "(by)" in tok2 and "order" in words:
                    col_match[idx] = []
                    continue
            elif tok == "TABLE" and len(words) > 2 :
                tok2,col2 = next_pattern_tok(raw_pattern_tokens,col_match,idx,q_tokens)
                if tok2 and "(by)" in tok2 and "order" in words:
                    table_match[idx] = []
                    continue
            elif tok == "AGG" and "COL" in words and idx + 1 < len(tokens) and col_match[idx] and col_match[idx+1]:
                for col in col_match[idx][0]:
                    if col not in col_match[idx+1][0]:
                        col_match[idx] = []
                        return tok
                return "COL"
            elif tok == "SJJS" and ("GR_JJS" in words or "SM_JJS" in words):
                continue
            elif tok == "SGRSM" and ("GR_GRSM" in words or "SM_GRSM" in words):
                continue
            elif "under" in words or "over" in words:
                for i in range(idx+1,len(raw_pattern_tokens)):
                    if "NUM," in raw_pattern_tokens[i][0] != "#":
                        return tok
                for i in range(idx+1,len(col_match)):
                    if col_match[i]:
                        for c in col_match[i][0]:
                            if schema.column_types[c] != "text":
                                return tok
                return "#"
            elif tok == "least" and idx + 1 < len(raw_pattern_tokens) and raw_pattern_tokens[idx + 1][0].startswith("SM_JJS"):
                return "SM_JJS"

            return tok
    return "*"



def pattern_recomand(sentence_ts,table_match,col_match,entt,db_match,schema,table_idxs,print_result=False):
    full_pattern_str = ""
    pattern_str = ""

    pattern_every_token = []
    raw_pattern_tokens = []
    raw_pattern_tokens2 = []


    for i, token in enumerate(sentence_ts.tokens):
        if not get_punctuation_word(sentence_ts.tokens,i) and  not table_match[i] and not col_match[i] and not db_match[i] and (token.text in DELETE_WORDS or token.text in [".","?",","] or (sentence_ts.tokens[i].tag_ == "DT" and  sentence_ts.tokens[i].text not in ["each"])) and (sentence_ts.tokens[i].text.islower() or i==0 or not sentence_ts.tokens[i].text.isalpha()) and sentence_ts.tokens[i].text not in ["and","or"]: # delete STOP WORD.
            if token.text in NEGATIVE_WORDS or token.lemma_ in NEGATIVE_WORDS:
                raw_pattern_tokens.append(("NOT", "NOT"))
            elif token.lower_ in {"that"} and i == 0:
                raw_pattern_tokens.append(("that", "that"))
            else:
                raw_pattern_tokens.append(("#", "#"))
            continue
        
        p_str, p_toks = pattern_word_guess(sentence_ts.tokens[i],table_match[i],col_match[i],table_match,col_match,entt[i],db_match[i],sentence_ts.tokens,schema,table_idxs,i,True)
        raw_pattern_tokens.append((p_str, p_toks))

    for i, (p_str, p_toks) in enumerate(raw_pattern_tokens):
        if p_str != "#":
            pattern_every_token.append(choose_possible_pattern_token(p_toks,sentence_ts.tokens,i,raw_pattern_tokens,table_match,col_match,sentence_ts.tokens,schema))
            pattern_str += pattern_every_token[-1]
            pattern_str += " "
            full_pattern_str += p_str + " | "
        else:
            pattern_every_token.append("#")
            full_pattern_str += "# | "
    
    if "COL" not in pattern_str and "AGG,COL" in full_pattern_str and pattern_every_token.count("AGG") == 1:
        pattern_str = pattern_str.replace("AGG","COL")
        pattern_every_token = ['COL' if i =='AGG' else i for i in pattern_every_token]

    if print_result:
        print(sentence_ts)
        print(full_pattern_str)
        print(pattern_str)
        print()
    return pattern_str,pattern_every_token,full_pattern_str


