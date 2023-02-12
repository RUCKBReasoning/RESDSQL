import json
from .TokenString import TokenString, get_spacy_tokenizer
from .Schema_Token import Schema_Token
from .stemmer import MyStemmer
lstem = MyStemmer()

# The STOP_WORDS is adapted from https://github.com/benbogin/spider-schema-gnn/blob/02f4ae43b891f41909215e889e37fbc084f982e1/semparse/contexts/spider_db_context.py
STOP_WORDS = {"", "'", "all", "being", "-", "over", "through", "yourselves", "its", "before",
              "hadn", "with", "had", ",", "should", "to", "only", "under", "ours", "has", "ought", "do",
              "them", "his", "than", "very", "cannot", "they", "not", "during", "yourself", "him",
              "nor", "did", "didn", "'ve", "this", "she", "each", "where", "because", "doing", "some", "we", "are",
              "further", "ourselves", "out", "what", "for", "weren", "does", "above", "between", "mustn", "?",
              "be", "hasn", "who", "were", "here", "shouldn", "let", "hers", "by", "both", "about", "couldn",
              "of", "could", "against", "isn", "or", "own", "into", "while", "whom", "down", "wasn", "your",
              "from", "her", "their", "aren", "there", "been", ".", "few", "too", "wouldn", "themselves",
              ":", "was", "until", "more", "himself", "on", "but", "don", "herself", "haven", "those", "he",
              "me", "myself", "these", "up", ";", "below", "'re", "can", "theirs", "my", "and", "would", "then",
              "is", "am", "it", "doesn", "an", "as", "itself", "at", "have", "in", "any", "if", "!",
              "again", "'ll", "no", "that", "when", "same", "how", "other", "which", "you", "many", "shan",
              "'t", "'s", "our", "after", "most", "'d", "such", "'m", "why", "a", "off", "i", "yours", "so",
              "the", "having", "once"
              ,"either","every",
              "name","id","names","ids"}

MUST_MORE_THAN_HALF = {"first","last","have","has","zero","single","one","two","three",\
    "four","five","six","seven","eight","nine","ten","eleven","twelve","thirteen",\
    "fourteen","fifteen","sixteen","seventeen","eighteen","nineteen","twenty","once"\
    ,"second","third","fourth","fifth","sixth","seventh","eighth","ninth","tenth","the","is","was","are","were","been", "am","a", "it", "an","its","their","ours","his",  "your", "her", "their", "and", 
    "this","that", "'",'"','of','some','any','many', "such","both","either","every","'re", "'ll", "'s", "'d", "'m", "'ve", "will", "would","could","can","do","did",",",":",".","?","(",")","in" }

def init_table_name(table_names):
    table_names = [ [i,tn] for i, tn in enumerate(table_names) ]
    table_names = sorted(table_names,key=lambda x:len(x[1].split(' ')),reverse=True)
    
class aa():
    def __init__(self):
        self.table_tokens_lemma_str = ["aa4","a0 bb cc","ff bb"]
        self.table_tokens_text_str = ["aa4","aa b0 cc","cc bb"]

def exact_match_idx(question_tokens,str_, type_=1,table=True):
    exact_match = []
    idx = -1
    while True: 
        idx = question_tokens.index(str_,idx+1,type_)
        if idx < 0:
            break
        elif idx == 0 and table:
            continue
        elif idx == 0 and question_tokens.tokens[0].text[0].isupper():
            continue
        elif idx > 0 and question_tokens.tokens[idx-1].text in [".","?"]:
            continue
        elif idx > 0 and question_tokens.tokens[idx].tag_ in ["JJR","JJS","RBR","RBS"] and question_tokens.tokens[idx].text != str_:
            continue
        exact_match.append(idx)
    return exact_match



def update_exact_match_tok(tok_indexes,table_idx,tok_len,exact_match,for_col_match_table_idx):
    for tok_index in tok_indexes:
        if exact_match[tok_index]:
            if exact_match[tok_index][1][0] > tok_len:
                continue

        for i in range(tok_len):
            if exact_match[i+tok_index]: # if there already contain data, reset data to delete it.
                if for_col_match_table_idx == -1:
                    exact_match[i+tok_index][0].append(table_idx)
                    exact_match[i+tok_index][1].append(tok_len)
                    exact_match[i+tok_index][2].append(1)
                    exact_match[i+tok_index][3].append(table_idx)
                else:
                    exact_match[i+tok_index][0] = [table_idx]
                    exact_match[i+tok_index][1] = [tok_len]
                    exact_match[i+tok_index][2] = [1]
                    exact_match[i+tok_index][3] = [table_idx]
            else:   # there is no data inside. Use append to add data.
                exact_match[i+tok_index].append([table_idx])
                exact_match[i+tok_index].append([tok_len])
                exact_match[i+tok_index].append([1])
                exact_match[i+tok_index].append([table_idx])
    return exact_match


def clean_table(full_match,question_tokens,schema,for_table_match=True,table_idx=0):
    for j,row in enumerate(full_match) :
        if row and 0 in row[2]:
            match_times=[1]*len(row[0]) # match times
            for i in range(len(row[0])):
                already = []
                tc = row[0][i]
                t  = row[3][i]
                for k,row2 in enumerate(full_match) :
                    if row2 and 0 in row2[2]:
                        if t in row2[3] and tc not in row2[0] and row2[0][row2[3].index(t)] not in already and abs(j-k)<=3:
                            match_times[i] += 1
                            already.append(row2[0][row2[3].index(t)])
            row[1] = match_times

    for j,row in enumerate(full_match):
        if row and 0 in row[2]:
            max_ = max(row[1])
            
            remove_others_cols = False
            if max_ > 1 and not for_table_match:
                for i in reversed(range(len(row[1]))):
                    if row[1][i] == max_:
                        col_end = schema.column_tokens_lemma_str[schema.tbl_col_idx_back[table_idx][row[3][i]]].split(" ")[-1]
                        if " " + col_end in question_tokens.lemma_ or col_end in ["name","id"]:
                            remove_others_cols= True

            for i in reversed(range(len(row[1]))):
                if row[2][i] == 1:
                    continue
                is_not_end_with = True
                if row[1][i] != max_ or (for_table_match and max_ == 1 and (question_tokens.tokens[j].lemma_ in ['list','name'] or question_tokens.tokens[j].lemma_ in STOP_WORDS)):
                    if for_table_match:
                        for tb in schema.table_tokens_lemma_str[row[3][i]].split(" | "):
                            if tb.endswith(question_tokens.tokens[j].lemma_):
                                is_not_end_with = False
                                break
                    else:
                        pass
                        if not remove_others_cols and "name" in schema.column_tokens_lemma_str[schema.tbl_col_idx_back[table_idx][row[3][i]]]:
                            is_not_end_with = False
                            
                if is_not_end_with and ((row[1][i] != max_ and row[1][i] == 1) or (for_table_match and max_ == 1 and (question_tokens.tokens[j].lemma_ in ['list','name'] or question_tokens.tokens[j].lemma_ in STOP_WORDS))):
                    del row[0][i]
                    del row[1][i]
                    del row[2][i]
                    del row[3][i]
                else:
                    row[0][i] = int(row[0][i])
    
    for k,row in enumerate(full_match): # remove the same rows
        if row and len(row[0]) == len(row[1]) and len(row[0]) == len(row[-1]):
            for j in range(len(row[0])):
                for i in reversed(range(len(row[0]))):
                    if i <= j:
                        break
                    if row[0][i] == row[0][j] and row[1][i] == row[1][j] and row[2][i] == row[2][j]:
                        if len(row) == 4:
                            if row[3][i] != row[3][j]:
                                continue
                            else:
                                del row[3][i]
                        del row[0][i]
                        del row[1][i]
                        del row[2][i]
    return full_match


def final_table_list(full_match):
    table_list = set()
    for j,row in enumerate(full_match):
        if row:
            for i in row[0]:
                table_list.add(int(i))
    return table_list



def update_partly_match_tok(tok_indexes,table_idx,tok_in_table_idx,tok_len,exact_match,real_len,previous_match_idx):

    new_table_idx = round(table_idx + tok_in_table_idx/1000,3)
    for tok_index in tok_indexes:
        if exact_match[tok_index]:
            if exact_match[tok_index][1][0] > real_len or (exact_match[tok_index][2][0] and False) or table_idx in exact_match[tok_index][0]:
                continue
        for i in range(tok_len):
            if exact_match[i+tok_index]:
                exact_match[i+tok_index][0].append(round(table_idx + previous_match_idx[len(previous_match_idx)+i-tok_len]/1000,3))
                exact_match[i+tok_index][1].append(1)
                exact_match[i+tok_index][2].append(0)
                exact_match[i+tok_index][3].append(table_idx)
            else:
                exact_match[i+tok_index].append([round(table_idx + previous_match_idx[len(previous_match_idx)+i-tok_len]/1000,3)])
                exact_match[i+tok_index].append([1])
                exact_match[i+tok_index].append([0])
                exact_match[i+tok_index].append([table_idx])
        
    return exact_match


def exact_match_table_name_plus_col(schema,question,question_tokens,table_name_list,full_match,match_type=1,table=True):
    for i, t_toks in enumerate(table_name_list):
        new_list = t_toks.split(' | ')
        for t_tok in new_list:
            if " " not in t_tok and " " + t_tok + " " in question:
                tok_indexes = exact_match_idx(question_tokens,t_tok,match_type,table=table)
                for tok_index in tok_indexes:
                    if i not in full_match[tok_index][1] and tok_index + 1 < len(question_tokens.tokens):
                        extra_add = False
                        for col1,col2 in zip(schema.table_col_lemma[i],schema.table_col_text[i]):
                            for c in col1.split(" | "):
                                if c == question_tokens.tokens[tok_index + 1].lemma_:
                                    extra_add = True
                                    break
                            if extra_add:
                                break
                            for c in col2.split(" | "):
                                if c == question_tokens.tokens[tok_index + 1].lemma_:
                                    extra_add = True
                                    break
                            if extra_add:
                                break
                        if  extra_add:
                            full_match[tok_index][0].append(i)
                            full_match[tok_index][1].append(1)
                            full_match[tok_index][2].append(1)
                            full_match[tok_index][3].append(i)
                            continue
    return full_match


def exact_match_table_name(question,question_tokens,table_name_list,full_match,match_type=1,table=True,table_idx=-1):
    for i, t_toks in enumerate(table_name_list):
        new_list = t_toks.split(' | ')
        for t_tok in new_list:
            if " " + t_tok + " " in question:
                tok_indexes = exact_match_idx(question_tokens,t_tok,match_type,table=table)
                full_match = update_exact_match_tok(tok_indexes,i,t_tok.count(" ")+1,full_match,table_idx)
    return full_match


def exact_stem_match_table_name(question,table_name_list,full_match,match_type=1,table=True,table_idx=-1):
    for i, t_toks in enumerate(table_name_list):
        new_list = t_toks.split(' | ')
        for t_tok in new_list:
            if " " in t_tok and " " + t_tok + " " in question:
                idx_t_tok = question.index(t_tok)
                a_l = question[1:idx_t_tok]
                tok_indexes = [a_l.count(" ")]
                full_match = update_exact_match_tok(tok_indexes,i,t_tok.count(" ")+1,full_match,table_idx)
    return full_match


def partly_match_table_name(question,question_tokens,table_name_list,full_match,match_type=1,table=True,modified_fun=None,table_name_for_col_match=[]):
    
    for i, t_toks in enumerate(table_name_list):
        if " " in t_toks:
            new_list = t_toks.split(' | ')
            for t_tok in new_list:
                if " " in t_tok:
                    too = t_tok.split(' ')
                    tok_len = 1
                    previous_match_idx = []
                    for j,t in enumerate(too):
                        if modified_fun:
                            t = modified_fun(t,question)
                        if (t not in table_name_for_col_match or j==len(too)-1) and t not in STOP_WORDS and " " + t + " " in question and not (t == "name" and t_tok in ["first name","last name"]):
                            tok_indexes = exact_match_idx(question_tokens,t,match_type,table=table)
                            previous_match_idx.append(j+1)
                            full_match = update_partly_match_tok(tok_indexes,i,j+1,1,full_match,tok_len,previous_match_idx)
                            tok_len += 1
                        elif (t not in table_name_for_col_match or j==len(too)-1) and t in STOP_WORDS and tok_len > 1 and j > 0 and " " + too[j-1] + " " + t + " " in question and not (t == "name" and t_tok in ["first name","last name"]):
                            tok_indexes = exact_match_idx(question_tokens,too[j-1] + " " + t,match_type,table=table)
                            previous_match_idx.append(j+1)
                            full_match = update_partly_match_tok(tok_indexes,i,j+1,2,full_match,tok_len,previous_match_idx)
                            tok_len += 1
    return full_match


def combined_exact_match_table_name(question,question_tokens,table_name_list,full_match,match_type=1,table=True,table_idx=-1,tname=None):
    for i, t_toks in enumerate(table_name_list):
        new_list = t_toks.split(' | ')
        for t_tok in new_list:
            if t_tok.count(" ") == 1:
                t = t_tok.replace(' ',"")
            elif t_tok.count(" ") == 0 and tname:
                t = tname + t_tok
            else:
                continue
            if  " " + t + " " in question:
                tok_indexes = exact_match_idx(question_tokens,t,match_type,table=table)
                full_match = update_exact_match_tok(tok_indexes,i,1,full_match,table_idx)
            else:
                if t_tok.count(" ") == 0 and ((len(t_tok) == 5 and t_tok.endswith("name")) or len(t_tok) == 3 and t_tok.endswith("id")):
                    startword = t_tok[0]
                    endword = t_tok[1:]
                    if  " " + endword + " " in question:
                        q_tokens = question.split(" ")
                        idx_end = q_tokens.index(endword)
                        if (idx_end > 2 and q_tokens[idx_end-1][0] == startword) :
                            tok_indexes = exact_match_idx(question_tokens,q_tokens[idx_end-1]+" "+endword,match_type,table=table)
                            full_match = update_exact_match_tok(tok_indexes,i,2,full_match,table_idx)
                        elif (idx_end + 3 < len(q_tokens) and q_tokens[idx_end+2][0] == startword and q_tokens[idx_end+1] == "of"):
                            tok_indexes = exact_match_idx(question_tokens,endword+" of "+q_tokens[idx_end+2],match_type,table=table)
                            full_match = update_exact_match_tok(tok_indexes,i,3,full_match,table_idx)
                else:
                    for j,t in enumerate(t_tok.split(' ')):
                        if len(t) >= 8 and t in question.replace(' ',""):
                            for k in range(3,len(t),1):
                                if " " + t[:k]+" "+t[k:] + " " in question:
                                    tok_indexes = exact_match_idx(question_tokens, t[:k]+" "+t[k:], match_type,table=table)
                                    full_match = update_exact_match_tok(tok_indexes,i,2,full_match,table_idx)
                                    break
    return full_match



def modified_exact_match_table_name_one_word(question_tokens,table_name_list,full_match,table=True):
    # lstem = nltk.stem.LancasterStemmer()
    for i, t_toks in enumerate(table_name_list):
        new_list = t_toks.split(' | ')
        for t_tok in new_list:
            if " " not in t_tok:
                t = lstem.stem(t_tok)
                for z, qtok in enumerate(question_tokens.tokens):
                    if t == lstem.stem(qtok.text):
                        if full_match[z]:
                            if full_match[z][1][0] >= 1:
                                continue
                        if z == 0 and table:
                            continue
                        elif z == 0 and question_tokens.tokens[0].text[0].isupper():
                            continue
                        elif z > 0 and question_tokens.tokens[z-1].text in [".","?"]:
                            continue

                        if full_match[z]: # if there already contain data, reset data to delete it.
                            full_match[z][0].append(i)
                            full_match[z][1].append(1)
                            full_match[z][2].append(0)
                            full_match[z][2].append(i)
                        else:   # there is no data inside. Use append to add data.
                            full_match[z].append([i])
                            full_match[z].append([1])
                            full_match[z].append([0])
                            full_match[z].append([i])
    return full_match



def modified_exact_match_table_name(question_tokens,table_name_list,full_match):
    def next_modified_word_match(tw,qtoks,start_idx):
        stop = start_idx+3 if start_idx+3 < len(qtoks) else len(qtoks)
        for zi in range(start_idx+1,stop,1):
            if tw == lstem.stem(qtoks[zi].text):
                return True,zi
        return False,-1
        
    for i, t_toks in enumerate(table_name_list):
        if " " in t_toks:
            new_list = t_toks.split(' | ')
            for t_tok in new_list:
                if " " in t_tok:
                    too = t_tok.split(' ')
                    list_ = []
                    for j,t in enumerate(too):
                        if t not in STOP_WORDS and j < len(too)-1:
                            t = lstem.stem(t)
                            tok_indexes = []
                            for z, qtok in enumerate(question_tokens.tokens):
                                if t == lstem.stem(qtok.text) and qtok.tag_ not in ["JJR","JJS","RBR","RBS"]:
                                    count = 1
                                    tok_indexes.append(z)
                                    offset = 1
                                    while(True):
                                        if j+offset >= len(too):
                                            break
                                        if too[j+offset] in STOP_WORDS:
                                            offset += 1
                                            continue
                                        tw = lstem.stem(too[j+offset])
                                        b_match, q_idx_match =  next_modified_word_match(tw,question_tokens.tokens,tok_indexes[-1])
                                        if b_match:
                                            tok_indexes.append(q_idx_match)
                                            offset += 1
                                        else:
                                            break
                                    if len(tok_indexes) > 1:
                                        # full_match = update_exact_match_tok(tok_indexes,i,len(tok_indexes),full_match)
                                        for qtok_index in tok_indexes:
                                            if full_match[qtok_index]:
                                                if full_match[qtok_index][1][0] >= len(tok_indexes):
                                                    continue
                                                if i in full_match[qtok_index][3]:
                                                    continue
                                            if qtok_index == 0:
                                                continue
                                            elif question_tokens.tokens[qtok_index-1].text in [".","?"]:
                                                continue

                                            if full_match[qtok_index]: # if there already contain data, reset data to delete it.
                                                full_match[qtok_index][0].append(i)
                                                full_match[qtok_index][1].append(len(tok_indexes))
                                                full_match[qtok_index][2].append(0)
                                                full_match[qtok_index][3].append(i)
                                            else:   # there is no data inside. Use append to add data.
                                                full_match[qtok_index].append([i])
                                                full_match[qtok_index].append([len(tok_indexes)])
                                                full_match[qtok_index].append([0])
                                                full_match[qtok_index].append([i])
                                    tok_indexes = []


    return full_match


def return_table_name(question_tokens,table_name_tokens):
    full_match  = [[] for i in question_tokens.tokens]

    table_match = []
    question_tokens.delete_suffix()
    question_lemma = " " + question_tokens.lemma_ + " "
    question_text = " " + question_tokens.text + " "
    # Exact Match:
    full_match = exact_match_table_name(question_lemma,question_tokens,table_name_tokens.table_tokens_lemma_str,full_match,match_type=2)
    
    full_match = exact_match_table_name(question_text,question_tokens,table_name_tokens.table_tokens_lemma_str,full_match)
    
    full_match = exact_match_table_name(question_text,question_tokens,table_name_tokens.table_tokens_text_str,full_match)
    
    full_match = exact_match_table_name_plus_col(table_name_tokens,question_lemma,question_tokens,table_name_tokens.table_tokens_lemma_str,full_match,match_type=2)
    

    
    # Partly Match:
    full_match = partly_match_table_name(question_lemma,question_tokens,table_name_tokens.table_tokens_lemma_str,full_match,match_type=2)
    
    full_match = partly_match_table_name(question_text,question_tokens,table_name_tokens.table_tokens_lemma_str,full_match)

    full_match = partly_match_table_name(question_text,question_tokens,table_name_tokens.table_tokens_text_str,full_match)

    # Modified exact match
    full_match = modified_exact_match_table_name_one_word(question_tokens,table_name_tokens.table_tokens_lemma_str,full_match)
    full_match = modified_exact_match_table_name(question_tokens,table_name_tokens.table_tokens_lemma_str,full_match)


    full_match = clean_table(full_match,question_tokens,table_name_tokens)
    full_match = clean_p_table_match(full_match,question_tokens,table_name_tokens)

    table_match = final_table_list(full_match)
    return table_match,full_match


def clean_star(full_match,schema):
    for j,row in enumerate(full_match):
        if row:
            for i in reversed(range(len(row[1]))):
                if schema.column_names_original[row[0][i]][1] == '*':
                    del row[0][i]
                    del row[1][i]
                    del row[2][i]
                    del row[3][i]
            if not row[0]:
                full_match[j] = []
    return full_match


def clean_p_col_match(full_match,question_tokens,schema):
    """
        We should use the original table file for here
    """
    for j,row in enumerate(full_match):
        if row:
            for i in reversed(range(len(row[1]))):
                if row[2][i] == 0 and row[1][i] / len(schema.column_tokens_lemma_str_tokens[row[0][i]]) <= 0.5:
                    if question_tokens.tokens[j].lemma_ in MUST_MORE_THAN_HALF or (" id" in schema.column_tokens_lemma_str[row[0][i]] and question_tokens.tokens[j].lemma_ != "id"):
                        if (" each " in question_tokens.lemma_ and len(row[0]) == 1) or (question_tokens.tokens[j].lemma_ == "first" and "last name" in question_tokens.lemma_)or (question_tokens.tokens[j].lemma_ == "last" and "first name" in question_tokens.lemma_):
                            continue
                        del row[0][i]
                        del row[1][i]
                        del row[2][i]
                        del row[3][i]
                    elif j + 1 < len(full_match) and question_tokens.tokens[j].lemma_ == "number" and question_tokens.tokens[j+1].lemma_ == "of":
                        del row[0][i]
                        del row[1][i]
                        del row[2][i]
                        del row[3][i]
                    elif  row[1][i] == 1 and question_tokens.tokens[j].text.isdigit():
                        del row[0][i]
                        del row[1][i]
                        del row[2][i]
                        del row[3][i]
            if not row[0]:
                full_match[j] = []
    return full_match


def clean_p_table_match(full_match,question_tokens,schema):
    """
        We should use the original table file for here
    """
    for j,row in enumerate(full_match):
        if row:
            for i in reversed(range(len(row[1]))):
                if row[2][i] == 0 and row[1][i] / len(schema.column_tokens_lemma_str_tokens[row[0][i]]) <= 0.5:
                    if question_tokens.tokens[j].lemma_ in MUST_MORE_THAN_HALF:
                        del row[0][i]
                        del row[1][i]
                        del row[2][i]
                        del row[3][i]
            if not row[0]:
                full_match[j] = []
    return full_match

def remove_ment_ship(t,question):
    if t.endswith("ment") or t.endswith("ship"):
        return t[:-4]
    elif "ship " in question:
        return t + "ship"
    elif "ment " in question:
        return t + "ment"
    elif "ments " in question:
        return t + "ments"
    return t

def return_column_match(question_tokens,table_name_tokens,table_idx,full_match = None,only_exact_match=False):
    STOP_WORDS.remove("names")
    STOP_WORDS.remove("name")
    STOP_WORDS.remove("id")
    STOP_WORDS.remove("ids")
    if full_match is None:
        full_match  = [[] for i in question_tokens.tokens]

    table_match = []
    question_lemma = " " + question_tokens.lemma_without_jjs_jjr() + " "
    question_text = " " + question_tokens.text + " "
    question_stem = " " + " ".join([lstem.stem(tok.text) for tok in question_tokens.tokens]) + " "
    # Exact Match:
    full_match = exact_match_table_name(question_lemma,question_tokens,table_name_tokens.tbl_col_tokens_lemma_str[table_idx],full_match,match_type=2,table=False,table_idx=table_idx)
    
    full_match = exact_match_table_name(question_text,question_tokens,table_name_tokens.tbl_col_tokens_lemma_str[table_idx],full_match,table=False,table_idx=table_idx)
    
    full_match = exact_match_table_name(question_text,question_tokens,table_name_tokens.tbl_col_tokens_text_str[table_idx],full_match,table=False,table_idx=table_idx)
    
    full_match = exact_stem_match_table_name(question_stem,table_name_tokens.tbl_col_tokens_stem_str[table_idx],full_match,table=True,table_idx=table_idx)
    
    full_match = combined_exact_match_table_name(question_lemma,question_tokens,table_name_tokens.tbl_col_tokens_lemma_str[table_idx],full_match,match_type=2,table=False,table_idx=table_idx,tname=table_name_tokens.table_tokens_lemma_str[table_idx])

    if not only_exact_match:
        # Partly Match:
        table_name_for_col_match = [table_name_tokens.table_tokens_lemma_str[table_idx].split(" | ")[0].split(" ")[-1],table_name_tokens.table_tokens_text_str[table_idx].split(" | ")[0].split(" ")[-1]]
        full_match = partly_match_table_name(question_lemma,question_tokens,table_name_tokens.tbl_col_tokens_lemma_str[table_idx],full_match,match_type=2,table=False,table_name_for_col_match=table_name_for_col_match)
        
        full_match = partly_match_table_name(question_text,question_tokens,table_name_tokens.tbl_col_tokens_lemma_str[table_idx],full_match,table=False,table_name_for_col_match=table_name_for_col_match)

        full_match = partly_match_table_name(question_text,question_tokens,table_name_tokens.tbl_col_tokens_text_str[table_idx],full_match,table=False,table_name_for_col_match=table_name_for_col_match)

        full_match = partly_match_table_name(question_lemma,question_tokens,table_name_tokens.tbl_col_tokens_lemma_str[table_idx],full_match,table=False,modified_fun=remove_ment_ship,table_name_for_col_match=table_name_for_col_match)

        # Modified exact match
        full_match = modified_exact_match_table_name_one_word(question_tokens,table_name_tokens.tbl_col_tokens_lemma_str[table_idx],full_match,table=False)
        full_match = modified_exact_match_table_name(question_tokens,table_name_tokens.tbl_col_tokens_lemma_str[table_idx],full_match)


    full_match = clean_table(full_match,question_tokens,table_name_tokens,False,table_idx)
    for fm in full_match:
        if fm:
            for i,f in enumerate(fm[0]):
                fm[0][i] = table_name_tokens.tbl_col_idx_back[table_idx][f]
            for i,f in enumerate(fm[3]):
                fm[3][i] = table_name_tokens.tbl_col_idx_back[table_idx][f]
    full_match = clean_star(full_match,table_name_tokens)
    STOP_WORDS.update({"name","id","names","ids"})
    full_match = clean_p_col_match(full_match,question_tokens,table_name_tokens)

    table_match = final_table_list(full_match)
    return table_match,full_match

