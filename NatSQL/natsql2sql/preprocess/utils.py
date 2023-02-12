import copy,re
from .match import STOP_WORDS,S_ADJ_WORD_DIRECTION
from .stemmer import MyStemmer


def is_float(s):
    s = str(s)
    if s.count('.') ==1:
        left = s.split('.')[0]
        right = s.split('.')[1]
        if right.isdigit():
            if left.count('-')==1 and left.startswith('-'):
                num = left.split['-'][-1]
                if num.isdigit():
                    return True
            elif left.isdigit():
                return True
    return False

def is_negative_digit(s):
    s = str(s)
    if s.startswith('-') and len(s) > 1:
        return s[1:].isdigit()
    return False

NUM = {"zero":0,"single":1,"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,"nine":9,"ten":10,"eleven":11,"twelve":12,"thirteen":13,"fourteen":14,"fifteen":15,"sixteen":16,"seventeen":17,"eighteen":18,"nineteen":19,"twenty":20,"once":1,"twice":2\
    ,"first":1,"second":2,"third":3,"fourth":4,"fifth":5,"sixth":6,"seventh":7,"eighth":8,"ninth":9,"tenth":10}

def str_is_num(s,recurrent=False):
    if not recurrent and type(s) == str and s.endswith("%") and str_is_num(s[:-1],True):
        return True
    return s.lower() in NUM.keys() or s.replace(",",'').isdigit() or is_float(s.replace(",",'')) or is_negative_digit(s.replace(",",'')) 

def str_is_date(s,all_tokens,s_idx):
    if re.fullmatch(r"^[A-Za-z]+$",s, flags=0):
        return None
    elif re.fullmatch(r"^([1][5-9]\d{2}|[2][0]\d{2})$",s, flags=0):
        if s_idx > 0 and all_tokens[s_idx-1].text in ["in","on","at","of","from"]:
            return "YEAR"

        for tok in all_tokens:
            if tok.lemma_ in ["young","old","late","early","elderly","new","late","previous",\
                "ancient","prior","quick","fast","slow","after","over","later","fast","elderly","rapid","advanced"\
                    "lengthy","before","early","under","short","happen","between","open","spring","fall","summer","winter","in","on","from"]:
                return "YEAR"
    elif is_float(s):
        return None
    elif re.fullmatch(r'((\d{4}((_|-|/){1}\d{1,2}){2})|(\d{1,2}(_|-|/)){2}(\d{4}|\d{2})){0,1}\s{0,1}(\d{2}(:\d{2}){1,2}){0,1}',s, flags=0):
        return "DATE"
    elif re.fullmatch(r'(\d{1,2}(st|nd|rd|th){0,1}(,|\s|-)){0,1}((J|j)(an|AN)(uary){0,1}|(F|f)(eb|EB)(ruary){0,1}|(M|m)(ar|AR)(ch){0,1}|(A|a)(pr|PR)(il){0,1}|(M|m)(ay|AY)|(J|j)(un|UN)(e){0,1}|(J|j)(ul|UL)(y){0,1}|(A|a)(ug|UG)(ust){0,1}|(S|s)(ep|EP)(tember){0,1}|(O|o)(ct|CT)(ober){0,1}|(N|n)(ov|OV)(ember){0,1}|(D|d)(ec|EC)(ember){0,1})(\s|,|-)(\d{1,2}(st|nd|rd|th){0,1}(\s|,){1,3}){0,1}(\d{4}|\d{2})',s, flags=0):
        return "DATE"
    if re.fullmatch(r"^(\d{1,2}:\d{1,2}(.){0,1}\d{0,2}(.){0,1}\d{0,2})$",s, flags=0):
        return "DATE"
    return None


def str_is_special_num(s):
    if len(s) < 4 or not s.isdigit():
        return False
    elif re.fullmatch(r"^([1][5-9]\d{2}|[2][0]\d{2})$",s, flags=0):
        return False
    elif s.endswith("00"):
        return False
    else:
        return True


def number_back(s):
    if s in NUM.keys():
        return str(NUM[s])
    return s



def look_for_table_idx(sub_q, list_idx, select_type, schema):
    """
    sub_q:
        sub question objs(contain all sub questin information)
    list_idx:
        the idx fo sub question who need table idxs
    find the table match idxs from the most close sub question.
    """
    if select_type == 1 and sub_q.sub_sequence_type[list_idx] <= select_type and 0 in sub_q.sub_sequence_type:
        table_list = sub_q.table_match_index(sub_q.sub_sequence_type.index(0), schema, False)
    elif sub_q.sub_sequence_type[list_idx] <= select_type:
        table_list = sub_q.get_select_table_index(list_idx, select_type, schema)
    else:
        table_list = sub_q.table_match_index(list_idx, schema)
    if not table_list and sub_q.sub_sequence_type[list_idx] <= select_type:
        table_list = []
        for i,t in enumerate(sub_q.sub_sequence_type):
            if i != list_idx and sub_q.sub_sequence_type[i] <= select_type:
                table_list.extend(sub_q.table_match_index(i, schema))
        if not table_list:
            table_list = sub_q.table_match_index(0, schema)
    if not table_list:
        i = 1
        while(True):
            table_list_l = sub_q.table_match_index(list_idx-i, schema)
            table_list_r = sub_q.table_match_index(list_idx+i, schema)
            if table_list_l == [-1] and table_list_r == [-1]:
                return [-1]
            elif table_list_l != [-1] and table_list_l:
                return table_list_l
            elif table_list_r != [-1] and table_list_r:
                return table_list_r
            i += 1
    if not table_list:
        table_list = [-1]
    return table_list



def look_for_closest_table_idx(table_idxs_list,start_idx):
    if table_idxs_list[start_idx]:
        return table_idxs_list[start_idx]
    left = start_idx
    right = start_idx
    while(True):
        left = left - 1
        right = right + 1
        if left < 0 and right >= len(table_idxs_list):
            return None
        if left >= 0 and table_idxs_list[left]:
            return table_idxs_list[left]
        elif right < len(table_idxs_list) and table_idxs_list[right]:
            return table_idxs_list[right]


def get_all_table(table_match,col_match,db_match,schema):
    re_match_idx = []
    for t in table_match:
        if t:
            re_match_idx.extend(t[0])
            
    for cm in col_match:
        if cm:
            for ci,c in enumerate(cm[0]):
                if cm[1][ci] > 1 or cm[2][ci] == 1:
                    re_match_idx.append(schema.column_names_original[c][0])
    for dbm in db_match:
        for cdb in dbm:
            re_match_idx.append(schema.column_names_original[cdb[0]][0])
    return list(set(re_match_idx))


def get_all_table_from_sq(sq,schema,col_appear_min_num=2):
    tbs = []
    tbs_c = dict()
    for ts,cs,dbs in zip(sq.table_match,sq.col_match,sq.db_match):
        for t in ts:
            if t:
                tbs.extend(t)
        
        for c in cs:
            if c:
                for cc in c[0]:
                    if schema.column_tokens_table_idx[cc] in tbs_c.keys():
                        tbs_c[schema.column_tokens_table_idx[cc]] += 1
                    else:
                        tbs_c[schema.column_tokens_table_idx[cc]] = 1
        for dbc in dbs:
            if dbc:
                for cc in dbc:
                    if schema.column_tokens_table_idx[cc[0]] in tbs_c.keys():
                        tbs_c[schema.column_tokens_table_idx[cc[0]]] += 1
                    else:
                        tbs_c[schema.column_tokens_table_idx[cc[0]]] = 1
    for tb in tbs_c.keys():
        if tbs_c[tb] >= col_appear_min_num:
            tbs.append(tb)
    return list(set(tbs))


def get_all_col_from_sq(sq,schema,table_idx=None):
    cols = []
    for cs,dbs in zip(sq.col_match,sq.db_match):
        for c in cs:
            if c:
                for cc in c[0]:
                    if not table_idx or schema.column_tokens_table_idx[cc] in table_idx:
                        cols.append(cc)
        for dbc in dbs:
            if dbc:
                for cc in dbc:
                    if not table_idx or schema.column_tokens_table_idx[cc[0]] in table_idx:
                        cols.append(cc)
    return cols



def construct_select_data(sentence, table_match, sentence_idxs, schema):
    """
    select: [column1, column2, ...]
    column: [words1, words2] # words is cut by the prep. Sometimes, words are table.
    words:  [word,word]      # word is string
    """
    def reset_element_list(element_sent,element_table,element_idx,layer_sent,layer_table,layer_idx):
        if element_sent:
            layer_sent.append(element_sent)
            layer_table.append(element_table)
            layer_idx.append(element_idx)
            element_sent = copy.deepcopy([])
            element_table = copy.deepcopy([])
            element_idx = copy.deepcopy([])
        return element_sent,element_table,element_idx,layer_sent,layer_table,layer_idx


    sentence_toks = sentence.split(" ")
    availble_sent = []
    availble_table = []
    original_sent = []
    original_table = []
    layer_sent = []
    layer_table = []
    element_sent = []
    element_table = []

    availble_idx = []
    original_idx = []
    layer_idx = []
    element_idx = []

    for tok,tb,i in zip(sentence_toks,table_match,sentence_idxs):
        if (i ==  sentence_idxs[0] or (i == sentence_idxs[1] and sentence_toks[0] in [',',"?",'.'])) and tok in ["give","show","list","find","sort",",","?",'.',"and","count","select","return"]:
            continue
        if tok in [",","and"] and i != sentence_idxs[-1]:
            element_sent,element_table,element_idx,layer_sent,layer_table,layer_idx = reset_element_list(element_sent,element_table,element_idx,layer_sent,layer_table,layer_idx)
            availble_sent.append(layer_sent)
            availble_table.append(layer_table)
            availble_idx.append(layer_idx)
            original_sent.append( [w for layer in layer_sent for w in layer])
            original_table.append([w for layer in layer_table for w in layer])
            original_idx.append([w for layer in layer_idx for w in layer])
            layer_sent = copy.deepcopy([])
            layer_table = copy.deepcopy([])
            layer_idx = copy.deepcopy([])
            continue
        elif tok in ["of","by","per","for","does","do","did"]:
            element_sent,element_table,element_idx,layer_sent,layer_table,layer_idx = reset_element_list(element_sent,element_table,element_idx,layer_sent,layer_table,layer_idx)
            continue
        elif tok.lower() not in STOP_WORDS or tok.lower() == "when":
            element_sent.append(tok)
            element_table.append(tb)
            element_idx.append(i)
    element_sent,element_table,element_idx,layer_sent,layer_table,layer_idx = reset_element_list(element_sent,element_table,element_idx,layer_sent,layer_table,layer_idx)
    if layer_sent:
        availble_sent.append(layer_sent)
        availble_table.append(layer_table)
        availble_idx.append(layer_idx)
        original_sent.append( [w for layer in layer_sent for w in layer])
        original_table.append([w for layer in layer_table for w in layer])
        original_idx.append([w for layer in layer_idx for w in layer])

    # here is for ID = 6. show the name and the release year of the song by the youngest singer .
    if availble_sent and len(availble_sent[-1]) == 3:
        for s,t in zip(availble_sent[-1][1],availble_table[-1][1]):
            if t:
                return availble_sent,availble_table,availble_idx, original_sent,original_table, original_idx
            if not schema.column_contain_word(s):
                return availble_sent,availble_table,availble_idx, original_sent,original_table, original_idx
        for i in range(len(availble_sent)-1):
            if len(availble_sent[i]) > 1:
                return availble_sent,availble_table,availble_idx, original_sent,original_table, original_idx
            for t in availble_table[i][0]:
                if t:
                    return availble_sent,availble_table,availble_idx, original_sent,original_table, original_idx
        for i in range(len(availble_sent)-1):
            availble_table[i].append(availble_table[-1][1])
            availble_sent[i].append(availble_sent[-1][1])
            availble_idx[i].append(availble_idx[-1][1])
            for w,t,idx in zip(original_sent[-1],original_table[-1],original_idx[-1]):
                if idx in availble_idx[-1][1]:
                    original_sent[i].append(w)
                    original_table[i].append(t)
                    original_idx[i].append(idx)
    return availble_sent,availble_table,availble_idx, original_sent,original_table, original_idx



lstem = MyStemmer()

def sgrsm_key(tok_grsm):
    key = None
    if tok_grsm.text in S_ADJ_WORD_DIRECTION.keys():
        key = tok_grsm.text
    elif tok_grsm.lemma_ in S_ADJ_WORD_DIRECTION.keys():
        key = tok_grsm.lemma_
    elif lstem.stem(tok_grsm.text) in S_ADJ_WORD_DIRECTION.keys():
        key = lstem.stem(tok_grsm.text)
    return key

def is_there_sgrsm_and_gr_or_sm(all_tokens,tok_grsm,start_idx):
    """
    find out the closest word that related to the SGRSM
    """
    key = sgrsm_key(tok_grsm)
    if key:
        tick_tok = False
        off = 1
        while True:
            if tick_tok:
                idx_now = start_idx + off
            else:
                idx_now = start_idx - off
            if start_idx - off < 0 and start_idx + off >= len(all_tokens):
                break
            if tick_tok:
                off += 1
            tick_tok = not tick_tok
            if idx_now < 0 or idx_now >= len(all_tokens):
                continue 
            t_tok = all_tokens[idx_now]
            lstem_tok = lstem.stem(t_tok.text)
            for s_adj in S_ADJ_WORD_DIRECTION[key]:
                if s_adj[0] == t_tok.lemma_ or s_adj[0] == lstem_tok or s_adj[0] == t_tok.text:
                    if s_adj[2]:
                        return "SM_"
                    else:
                        return "GR_"
    return None

def sjjs_table(tok_grsm, table_col_lemma, table_col_idx):
    key = sgrsm_key(tok_grsm)
    re_list_idx = []
    re_list_dir = None
    if key:
        for s_adj in S_ADJ_WORD_DIRECTION[key]:
            for col,i in zip(table_col_lemma,table_col_idx):
                for ct in col.split(" "):
                    if s_adj[0] == ct:
                        if s_adj[2]:
                            grsm = "SM_"
                        else:
                            grsm = "GR_"
                        if re_list_dir and re_list_dir != grsm:
                            return [],"ERROR"
                        re_list_dir = grsm
                        re_list_idx.append(i)
                        if key in ["young","old"]:
                            return re_list_idx,re_list_dir
    return re_list_idx,re_list_dir



def get_punctuation_word(question_tokens,start_idx,only_bool=True,punct=['"',"'"]):
    num_left = 0
    num_right = 0
    for i in range(start_idx+1,len(question_tokens),1):
        if question_tokens[i].text in punct:
            num_right += 1
    for i in range(0,start_idx,1):
        if question_tokens[i].text in punct:
            num_left += 1
    if num_left and num_right:
        if (num_left % 2 == 0) and question_tokens[start_idx].text in punct and (num_right % 2 == 1):
            pass
        elif (num_right % 2 == 1): # I use this simple dangerous one to instead of the previous one.
            for i in range(start_idx,len(question_tokens),1):
                if question_tokens[i].text in punct:
                    num_right = i - 1
                    break
            for i in range(start_idx,0,-1):
                if question_tokens[i].text in punct:
                    num_left = i + 1
                    break
            if num_right - num_left <= 6 or punct!=['"',"'"] or num_right + 3 == len(question_tokens):
                if only_bool:
                    return True
                else:
                    return True,[num_left,num_right]
    if only_bool:
        return False
    else:
        return False,[0,0]