import json,editdistance
import copy
import re
from .TokenString import get_spacy_tokenizer,TokenString,SToken
from .table_match import return_table_name,return_column_match
from .sq import SubQuestion,QuestionSQL
from .others_pattern import pattern_reconize,pattern_recomand,get_col_from_related_word,get_AWD_column
from .utils import look_for_table_idx,str_is_date,str_is_num,get_all_table,get_punctuation_word
from .Schema_Token import Schema_Token
from .pattern_question_type import *
from .match import S_ADJ_WORD_DIRECTION,ABSOLUTELY_GRSM_DICT,A_ADJ_WORD_DIRECTION,SELECT_FIRST_WORD,SYNONYM
from .pattern_question_type import PATTERNS_TOKS,PATTERN_FUN
from .db_match import DBEngine,get_database_string
from .stemmer import MyStemmer
from .match import ABSOLUTELY_GRSM_DICT,NEGATIVE_WORDS
from .db_match import datebase_match_tables
from .match import COUNTRYS_DICT

NO_BREAK = 0
BREAK_PREP = 1
BREAK_SELECT= 2
BREAK_RELCL = 3
BREAK_ORDERBY = 4
lstem = MyStemmer()

STOP_WORDS = {"", "'", "all", "being", "-",  "through", "yourselves", "its", 
              "hadn", "with", "had", ",", "should", "to", "only", "ours", "has", "ought", "do",
              "them", "his", "very", "cannot", "they", "not",  "yourself", "him",
              "did", "didn", "'ve", "this", "she", "each", "where", "because", "doing", "some", "we", "are",
              "further", "ourselves", "out", "what", "for", "weren", "does", "mustn", "?",
              "be", "hasn", "who", "were", "here", "shouldn", "let", "hers", "by", "both", "about", "couldn",
              "of", "could", "isn", "or", "own", "while", "whom", "down", "wasn", "your",
              "from", "her", "their", "aren", "there", "been", ".", "few", "too", "wouldn", "themselves",
              ":", "was", "until",  "himself", "on", "but", "don", "herself", "haven", "those", "he",
              "me", "myself", "these", "up", ";", "'re", "can", "theirs", "my", "and", "would", "then",
              "is", "am", "it", "doesn", "an", "as", "itself", "at", "have", "any", "if", "!",
              "again", "'ll", "no", "that", "when", "same", "how", "other", "which", "you", "many", "shan",
              "'t", "'s", "our", "'d", "such", "'m", "why", "a", "off", "i", "yours", "so",
              "the", "having", "once"
              ,"either","every"}




def one_group_in_ahead(all_tokens,token,father_idx,offset):
    next_token = [token]
    valid_len = 0
    start = False
    break_check = True
    while True:
        tok_now = next_token[0]
        children = list(tok_now.children)
        del next_token[0]
        for child in children:
            next_token.append(child)
        if tok_now.i == 0 or (tok_now.i > 2 and all_tokens[tok_now.i-1-offset].text == ','):
            start = True
        elif tok_now.i > father_idx:
            break_check = False
            break
        if len(children) == 0 and len(next_token) == 0:
            if not start and tok_now.i == 1 and all_tokens[0].lemma_ == "which":
                start = True
            break
    return start and break_check

def token_list_to_one(token_list,old_type):
    """
    change the token list type from 0 to 1. This is for:

    from:
    Which year has most number of concerts ?
    [1, 1, 0, 0, 0, 0, 0, 0]
    ['1 Which', '1 year', '0 has', '0 most', '0 number', '0 of', '0 concerts', '0 ?']

    to:
    Which year has most number of concerts ?
    [0, 0, 1, 1, 1, 1, 1, 1]
    ['0 Which', '0 year', '1 has', '1 most', '1 number', '1 of', '1 concerts', '1 ?']
    """
    for tl in token_list:
        tl[1] = tl[1] + 1
    return 0,token_list


def check_valid_words(token,table_match,min_valid_len=1):
    """
    check how many valid words in the sentence. For example:
    what is dose it weigh ? Here only contain one valid word which is weigh.
    """
    next_token = [token]
    valid_len = 0
    while True:
        tok_now = next_token[0]
        children = list(tok_now.children)
        del next_token[0]
        for child in children:
            next_token.append(child)
        if tok_now.lower_ not in STOP_WORDS and not table_match[tok_now.i]:
            valid_len += 1
            if valid_len >= min_valid_len:
                return True
        if len(children) == 0 and len(next_token) == 0:
            return False
    return False

def check_continue_sentence(token,min_sent_len=3):
    """
    Check whether it is a continue sentence that start form the token.
    It can also be used to check the one_group_in_ahead. But I write different function.
    """
    next_token = [token]
    list_ = []
    uncount_token_len = 0
    while True:
        tok_now = next_token[0]
        children = list(tok_now.children)
        del next_token[0]
        for child in children:
            next_token.append(child)
        list_.append(tok_now.i)
        if tok_now.text in ["and","but","or","'",",",".","?"]:
            uncount_token_len += 1
        if len(children) == 0 and len(next_token) == 0:
            break
    list_.sort()
    if len(list_) >= min_sent_len + uncount_token_len:
        len_ = list_[-1] - list_[0] + 1
        if len_ == len(list_):
            return True
    return False

def check_behind_children_len(token):
    children = list(token.children)
    if not children:
        return 0
    count = 0
    final_child = None
    for child in children:
        if child.i > token.i:
            count += 1
            final_child = child
    if count == 1 and final_child and final_child.text in STOP_WORDS and len(list(final_child.children))==0:
        return 0
    return count


def prep_table_break(token,table_match):
    next_token = [c for c in token.children]
    valid_len = 0
    table_num = set()
    there_is_break = False
    while True:
        if len(next_token) == 0:
            if there_is_break:
                return True
            return False
        tok_now = next_token[0]
        children = list(tok_now.children)
        del next_token[0]
        if tok_now.text not in STOP_WORDS and not table_match[tok_now.i]:
            return False
        for i, child in enumerate(children) :
            
            if table_match[child.i]:
                table_num.add(table_match[child.i][0][0])
            if len(table_num) > 1:
                return False
            if child.dep_ == 'prep' and child.text not in ['of','for','by','between','than', 'per'] and check_behind_children_len(child):
                there_is_break = True
                continue
            elif child.dep_ == 'relcl':
                there_is_break = True
                continue
            elif child.text in STOP_WORDS or table_match[child.i]:
                pass
            else:
                return False
            next_token.append(child)
    return False


def special_case_for_prep(father_token):
    if father_token.text in[ 'ordered','order','sort','sorted']:
        children = " "
        for c in father_token.children:
            children += c.text + " "
        if " from " in children and " to " in children:
            return True
    return False



def check_break(sentence,token,father_type,father_idx,token_list,table_match,offset,sent_idx,sent_num):
    all_tokens = list(sentence)
    if (token.dep_ == 'prep' and token.text not in ['of','for','by','between','than', 'per'] and check_behind_children_len(token) and not special_case_for_prep(all_tokens[father_idx-offset])) :
        if father_type >= 1:
            if check_break_before(token_list,father_type,table_match) and check_valid_words(token,table_match,1):
                return BREAK_PREP
            return NO_BREAK
        elif prep_table_break(token,table_match) and token.i > father_idx:
            return NO_BREAK
        return BREAK_PREP
    elif token.dep_ in ['relcl','advcl']:
        return BREAK_RELCL
    elif token.dep_ == 'acl':
        if len(list(token.children)) == 0:
            return NO_BREAK
        if father_type >= 1:
            if check_break_before(token_list,father_type,table_match):
                return BREAK_RELCL
            return NO_BREAK
        return BREAK_RELCL
    elif token.dep_ in ['nsubj','npadvmod','csubj','nsubjpass'] and token.text != 'List' and one_group_in_ahead(all_tokens,token,father_idx,offset):
        if len(list(token.children)) == 0 and sent_idx == 0 and sent_num == 2:
            return BREAK_SELECT
        if len(list(token.children)) == 0 or (token.dep_ == 'csubj' and  not check_valid_words(token,table_match,1)):
            return NO_BREAK
        return BREAK_SELECT
    elif father_type >= 1 and token.dep_ == 'conj' and (check_valid_words(token,table_match,2) or check_continue_sentence(token)):
        return BREAK_RELCL
    elif token.text == 'ordered' and token.i + 2 - offset < len(all_tokens) and all_tokens[token.i + 1 - offset].text == 'by':
        return BREAK_ORDERBY
    return NO_BREAK


def check_break_before(token_list,val,table_match):
    count = 0
    num_tok = 0
    for tok in token_list :
        if tok[1] == val:
            for t in tok[0].text.split(' '):
                if t.lower() not in STOP_WORDS and not table_match[tok[0].i] and tok[0].dep_ != 'prep':
                    count += 1
                if table_match[tok[0].i]:
                    count += 0.5
                num_tok += 1
    if count >= 1 and num_tok > 1:
        return True
    return False




def merge_punctuation(token_list):
    for i,tok in enumerate(token_list):
        if tok[0].text in ['"',"'"] and i > 1:
            count = 0
            for j in range(i-1): 
                if token_list[j][0].text in ['"',"'"]:
                    count += 1
            if count > 0 and count % 2 != 0: 
                for j in range(i-1,-1,-1):
                    if token_list[j][0].text in ['"',"'"]:
                        if token_list[j][1] != token_list[i][1]:
                            token_list[i][1] = token_list[j][1]
                        break
    return token_list



def merge_punctuation_2(sentence,sent_data):
    c1 = sentence.lower_.count(" ' ")
    c2 = sentence.lower_.count(' " ')
    if (c2 and c2 % 2 == 0) or (c1 and c1 % 2 == 0):
        for i,tok in enumerate(sentence):
            if tok.text in ['"',"'"] and i > 1:
                count = 0
                for j in range(i-1): 
                    if sentence[j].text in ['"',"'"]:
                        count += 1
                if count > 0 and count % 2 != 0: 
                    for j in range(i-1,-1,-1):
                        if sentence[j].text in ['"',"'"]:
                            c = None
                            for target in sent_data:
                                for k,t in enumerate(target) :
                                    if t['idx'] == i:
                                        c = t
                                        del target[k]
                                        break
                                if c:
                                    break
                            c = None
                            for target in sent_data:
                                for t in target:
                                    if t['idx'] == j:
                                        c = copy.deepcopy(t)
                                        c['idx'] = i
                                        target.append(c)
                                        break
                                if c:
                                    break
                            break
    return sent_data


def sentence_cut(sentence,table_match,offset,sent_idx,sent_num):
    tokens = [token for token in sentence]
    root = [token for token in sentence if token.head == token]
    if len(root) > 1:
        return None
    root = root[0]

    type_ = 0
    idx = 0
    next_token = [[root,0]]
    token_list = [[root,type_]]
    select_break_times = 0
    break_several_times = 0
    while True:
        if len(next_token) == 0:
            break
        next_ = next_token[0][0]
        father_idx = next_token[0][1]
        del next_token[0]
        for child in next_.children:

            idx += 1
            next_token.append([child,idx])
            break_type = check_break(sentence,child,token_list[father_idx][1], token_list[father_idx][0].i, token_list,table_match, offset,sent_idx,sent_num)

            if break_type in [BREAK_RELCL,BREAK_PREP,BREAK_ORDERBY]:
                type_ = token_list[father_idx][1] + 1 + break_several_times + select_break_times
                break_several_times += 1
            elif break_type == NO_BREAK:
                type_ = token_list[father_idx][1]
            elif break_type == BREAK_SELECT:
                type_, token_list = token_list_to_one(token_list,token_list[father_idx][1])
                select_break_times += 1
            token_list.append([child,type_])
    
    token_list = sorted(token_list,key=lambda x:x[0].i)
    token_list = merge_punctuation(token_list)
    return token_list


def reshap_token(token_list):
    reshap_list=[]
    for tl in token_list:
        reshap_list.extend([tl[1]]*(tl[0].text.count(" ")+1))
    set_reshap_list = list(set(reshap_list))
    set_reshap_list.sort()
    new_v = dict()
    if -1 in set_reshap_list:
        for idx_l,val in zip(set_reshap_list,range(len(set_reshap_list))):
            new_v[idx_l] = val
    else:
        for idx_l,val in zip(set_reshap_list,range(len(set_reshap_list))):
            new_v[idx_l] = val+1
    for i in range(len(reshap_list)):
        reshap_list[i] = new_v[reshap_list[i]]
    return reshap_list


def there_are_seperate_select(token_list):
    first_0 = False
    first_1 = False
    other_0 = 0
    second_idx = 0
    for tok in token_list:
        if tok[1] == 0:
            first_0 = True
            if first_0 and first_1:
                if not second_idx:
                    second_idx = tok[0].i
                other_0 += 1
        else:
            if first_0:
                first_1 = True
    if other_0 > 1:
        return True,second_idx
    return False,second_idx

def re_reconize_0_type(token_list):
    there_is_second,second_idx = there_are_seperate_select(token_list)
    if there_is_second:
        for i in range(second_idx,len(token_list),1):
            if token_list[i][0].text not in [',','and','also']:
                if token_list[i][0].lower_ in SELECT_FIRST_WORD:
                    if token_list[i][0].lower_ == 'how':
                        if i + 1 >= len(token_list) or token_list[i+1][0].lower_ not in ["much","many",'old']:
                            return False
                    return True
    return False


def re_set_select_value(token_list,start):
    def reset(token_list,start,offset=1,reset_original_v=0):
        new_type = token_list[start-1][1] + offset
        for j in range(start,len(token_list),1):
            if token_list[j][1] == reset_original_v:
                token_list[j][1] = new_type
            else:
                break
        return token_list

    # check punctuation ' xxxx ':
    count_pun = 0
    if token_list[start][0].text in ["'",'"']:
        for i in range(start,len(token_list),1):
            if token_list[i][0].text in ["'",'"']:
                count_pun += 1
        if count_pun > 0 and count_pun % 2 == 0:
            return reset(token_list,start,offset=0)
    else:
        for i in range(start,len(token_list),1):
            if token_list[i][0].text in ["'",'"']:
                count_pun += 1
        if count_pun > 0 and count_pun % 2 == 1:
            count_pun = 0
            for i in range(0,start,1):
                if token_list[i][0].text in ["'",'"']:
                    count_pun += 1
            if count_pun > 0 and count_pun % 2 == 1:
                return reset(token_list,start,offset=0)
    
     # check Preposition:
    if start > 5 and start + 2 < len(token_list) and token_list[start-2][0].text == 'in' and token_list[start][0].text == 'for' and token_list[start-3][1] == 0 and token_list[start-2][1] != token_list[start+2][1] and token_list[start+2][1] != token_list[start][1] and token_list[start-2][1] != 0:
        return reset(token_list,start-2,offset=0,reset_original_v=token_list[start-2][1])

    if (token_list[start][0].tag_ == 'IN' or token_list[start][0].dep_ == 'preconj') and token_list[start][0].text not in ['and','or']:
        if token_list[start][0].text == "than":
            return reset(token_list,start,offset=0)
        return reset(token_list,start)
    for i in range(start,len(token_list),1):
        if token_list[i][1] != 0:
            break
        if token_list[i][0].tag_ == 'IN' and token_list[i][0].text not in ['of','for','by','in','per','and','or']:
            return reset(token_list,start)

    # check new sentence:
    if token_list[start-1][0].text in ['.','?'] or token_list[start][0].text in ['.','?']:
        return reset(token_list,start)

    # check verb:
    for i in range(start,len(token_list),1):
        if token_list[i][1] != 0:
            break
        if token_list[i][0].tag_.startswith('VB') or token_list[i][0].text in ['sort','order']:
            return reset(token_list,start)

    return token_list
    

def re_analyse_sentence(token_list):
    yes,start = there_are_seperate_select(token_list)
    if yes and not re_reconize_0_type(token_list) and start > 0:
        return re_set_select_value(token_list,start)
    return token_list


def sentence_cut_len(token_list,sentence_split):
    len_count = []
    for i,tok in enumerate(token_list):
        if not len_count or i in sentence_split:
            len_count.append([tok[1],0,i])
        if tok[1] == len_count[-1][0]:
            len_count[-1][1] += 1
        else:
            len_count.append([tok[1],1,i])
    return len_count

def correct_121_pattern(token_list,sentence_split):
    len_count = sentence_cut_len(token_list,sentence_split)
    
    for cor in range(2,4,1):#(1) 1,1,2,2,1,1 -> 1,1,2,2,2,2 #(2) 3,3,4,4,3,3 -> 3,3,4,4,4,4 #(3) 0,0,1,1,0,0 -> 0,0,1,1,1,1
        for i,l in enumerate(len_count):
            if l[0] == cor: # check type
                if i >= 1 and i < len(len_count) - 1: # not the first and last one
                    if (l[1] <= 2 and len_count[i-1][0] == cor - 1 and len_count[i+1][0] == cor - 1)\
                        or (l[1] <= 3 and len_count[i-1][0] == cor - 1 and len_count[i+1][1] <= 2 and len_count[i+1][0] == cor - 1):
                        k = len_count[i+1][2]
                        while True:
                            if k >= len(token_list) or token_list[k][1] != cor-1:
                                break
                            token_list[k][1] = cor
                            k += 1
                        return token_list
    
    for new_v in [2,1]:
        for i,l in enumerate(len_count):
            # 2222,1,2222 or 2222,11,2222 -> 2222222222222
            # 1111,0,1111 or 1111,00,1111 -> 1111111111111
            if l[0] == new_v-1:
                if i >= 1 and i < len(len_count) - 1:
                    if (l[1] <= 2 and len_count[i-1][0] == new_v and len_count[i+1][0] == new_v):
                        for i in range(l[2],len_count[i+1][2],1):
                            token_list[i][1] = new_v
                        return token_list

            # 0,1,2... --> 0,2,2...
            # 2,1,3... --> 2,3,3,
            if l[1] == 1 and l[0] == 1 and i >= 1 and i < len(len_count) - 1:
                if len_count[i-1][0] != len_count[i+1][0]:
                    for j in range(l[2],len_count[i+1][2],1):
                        token_list[j][1] = len_count[i+1][0] if len_count[i+1][0] > 0 else len_count[i-1][0]
                    return token_list
            
            # 0,1,1,2... --> 0,2,2,2,...
            # 0,1,1,3... --> 0,3,3,3,...
            if l[1] == 2 and l[0] == 1 and i >= 1 and i < len(len_count) - 1:
                if token_list[l[2]][0].lemma_ in ["that","who","order","sort","be"] and len_count[i-1][0] == 0 and len_count[i+1][0] >= 2:
                    for j in range(l[2],len_count[i+1][2],1):
                        token_list[j][1] = len_count[i+1][0]
                    return token_list

            # 0,0,0,1,1,0. --> 0,0,0,1,1,1,1
            if l[1] == 2 and l[0] == 0 and i >= 2 and token_list[l[2]+1][0].text in [".","?"]:
                for j in range(l[2],l[2]+l[1],1):
                    token_list[j][1] = len_count[i-1][0]
                return token_list
    return token_list

def correct_special_pattern(token_list,question,offset_list):
    def only_how_many(token_list):
        if token_list[0][0].text == "How" and token_list[1][0].text == "many" and token_list[0][1] == 0 and token_list[1][1] == 0 and token_list[2][1] == 1:
            for i,tok in enumerate(token_list):
                if tok[1] > 1:
                     break
                else:
                    token_list[i][1] = 0
        return token_list
    def punctuation_for(token_list,question):
        if " , for " in question:
            for i in range(1,len(token_list)-1,1):
                if token_list[i][0].text == "," and token_list[i+1][0].text == "for" and token_list[i][1] == 0 and token_list[i+1][1] == 0:
                    next_ = 0
                    for j in range(i,len(token_list),1): # get next type value
                        if token_list[j][1] != 0:
                            next_ = token_list[j][1]
                            break
                    for j in range(i,len(token_list),1):
                        if token_list[j][1] != 0:
                            break
                        else:
                            token_list[j][1] = next_
        return token_list
    def correct_if_final_one_is_different(token_list):
        if len(token_list) > 5  and token_list[-2][1] != token_list[-3][1]:
            token_list[-2][1] = token_list[-3][1]
        return token_list
    def reset_sort_type(token_list,offset_list):
        if len(offset_list) == 2 and token_list[0][0].lower_ == "sort":
            for i in range(0,offset_list[0]):
                if token_list[i][1] != 0:
                    break
                else:
                    token_list[i][1] = 1
            for i in range(offset_list[0],offset_list[1]):
                if token_list[i][1] == 0:
                    break
            if i == offset_list[1]-1:
                for i in range(offset_list[0],offset_list[1]):
                    token_list[i][1] = 0

        elif len(offset_list) == 2 and token_list[offset_list[0]][0].lower_ == "sort":
            for i in range(offset_list[0],offset_list[1]):
                if token_list[i][1] != 0:
                    break
                else:
                    token_list[i][1] = 1
        elif len(offset_list) == 1 and token_list[0][0].lower_ == "sort" and token_list[0][1] == 0 and token_list[0][-1] == 0:
            for i,tok in enumerate(token_list):
                if tok[1] != 0:
                    for j in range(i,offset_list[0]):
                        if token_list[j][1] == 0:
                            for k in range(0,offset_list[0]):
                                if token_list[k][1] != 0:
                                    break
                                else:
                                    token_list[k][1] = 1
                            break
                    break

        return token_list

        
    token_list = only_how_many(token_list)
    token_list = punctuation_for(token_list,question)
    token_list = correct_if_final_one_is_different(token_list)
    token_list = reset_sort_type(token_list,offset_list)
    return token_list


def final_correct_special_pattern(token_list):
    def and_for_next_condition(token_list):
        for i,tok in enumerate(token_list):
            if (tok[0].text == "and" and i > 3 and i + 2 < len(token_list) and not (token_list[i-2][0].text == "between" or token_list[i-3][0].text == "between")) and tok[1] > 0 and token_list[i-1][1] == tok[1]:
                count = 0
                for j in range(i+1,len(token_list)-1,1):
                    if token_list[j][1] <= 0:
                        break
                    if token_list[j][1] != tok[1]:
                        if j - i == 1:
                            break
                        for k in range(i+1,j,1):
                            token_list[k][1] = token_list[j][1]
                        break
                    if tok[0].text not in STOP_WORDS:
                        count += 1
                    if count >= 2:
                        break
                if token_list[i-2][1] != token_list[i-1][1] and token_list[i-1][1] == token_list[i][1] and token_list[i][1] != token_list[i+1][1]:
                    token_list[i][1] = token_list[i+1][1]
                    token_list[i-1][1] = token_list[i+1][1]
                elif token_list[i-2][1] == token_list[i-1][1] and token_list[i-1][1] != token_list[i][1] and token_list[i][1] != token_list[i+1][1]:
                    token_list[i][1] = token_list[i+1][1]
                    token_list[i-1][1] = token_list[i+1][1]
            elif tok[0].text == "but" and i + 2 < len(token_list) and tok[1] !=  token_list[i+1][1]:
                token_list[i][1] = token_list[i+1][1]
            elif tok[0].text in ["?","."] and i + 3 < len(token_list) and tok[1] == token_list[i+1][1] :
                for j in range(0,i+1,1):
                    token_list[j][1] = token_list[j][1] + 1
        return token_list
    def repair_disjoin_phase(token_list):
        for i,tok in enumerate(token_list):
            if (tok[0].lemma_ in ["start","end", "participate", "serve", "same"] and i + 2 < len(token_list) and token_list[i+1][0].text in ["from","on","in","as"] and tok[1] != token_list[i+1][1]) or \
                (tok[0].lemma_ in ["prior","equal","assign"] and i + 2 < len(token_list) and token_list[i+1][0].text == "to" and tok[1] != token_list[i+1][1]) or \
                (tok[0].lemma_ == "ascend" and i + 2 < len(token_list) and token_list[i+1][0].text == "order" and tok[1] != token_list[i+1][1]) or \
                (tok[0].lemma_ == "descend" and i + 2 < len(token_list) and token_list[i+1][0].text == "order" and tok[1] != token_list[i+1][1]) or \
                (tok[0].lemma_ == "be" and i + 2 < len(token_list) and token_list[i+1][0].lemma_ in ABSOLUTELY_GRSM_DICT and tok[1] != token_list[i+1][1]) or \
                (tok[0].lemma_ == "belong" and i + 2 < len(token_list) and token_list[i+1][0].text == "to" and tok[1] != token_list[i+1][1]) or \
                (tok[0].lemma_ == "locate" and i + 2 < len(token_list) and token_list[i+1][0].text in ["in","at"] and tok[1] != token_list[i+1][1]) or \
               (tok[0].lemma_ == "long" and i > 2 and i + 2 < len(token_list) and token_list[i+1][0].text == "as" and token_list[i-1][0].text == "as" and tok[1] != token_list[i+1][1]) or \
                ((tok[0].lemma_ == "than" or str_is_num(tok[0].text) or tok[0].lemma_ in ["equal","in"]) and i >= 2 and i + 2 < len(token_list) and token_list[i+1][0].text == "or" and tok[1] != token_list[i+1][1] and (tok[0].lemma_ == "than" or token_list[i+2][0].lemma_ in ABSOLUTELY_GRSM_DICT or token_list[i+2][0].lemma_ in A_ADJ_WORD_DIRECTION) ) or \
                (tok[0].lemma_ == "or" and i > 2 and i + 2 < len(token_list) and token_list[i-1][0].text == "than" and token_list[i+1][0].text in ["equal","in"] and tok[1] != token_list[i+1][1]) or\
                (tok[0].lemma_ in ["and","to"] and i > 3  and i + 2 <= len(token_list) and token_list[i-1][0].text not in ["and","or"] and token_list[i-2][0].text not in ["and","or"] and token_list[i-3][0].text not in ["and","or"] and tok[1] != token_list[i+1][1] and (token_list[i-1][0].text  == "between" or token_list[i-2][0].text  == "between" or token_list[i-3][0].text == "between")) or \
                (i + 2 <= len(token_list) and token_list[i+1][0].text in ["and","to"] and i > 3 and  token_list[i-1][0].text not in ["and","or"] and token_list[i-2][0].text not in ["and","or"] and token_list[i-3][0].text not in ["and","or"] and tok[1] != token_list[i+1][1] and (token_list[i-1][0].text  == "between" or token_list[i-2][0].text  == "between" or token_list[i-3][0].text == "between")) or \
                (tok[0].lemma_ == "to" and i > 3  and i + 2 <= len(token_list) and token_list[i-1][0].text not in ["and","or"] and token_list[i-2][0].text not in ["and","or"] and token_list[i-3][0].text not in ["and","or"] and tok[1] != token_list[i+1][1] and (token_list[i-1][0].text  in ["range","from"] or token_list[i-2][0].text  in ["range","from"] or token_list[i-3][0].text in ["range","from"])) or \
                (i + 2 <= len(token_list) and token_list[i+1][0].text == "to" and i > 3 and  token_list[i-1][0].text not in ["and","or"] and token_list[i-2][0].text not in ["and","or"] and token_list[i-3][0].text not in ["and","or"] and tok[1] != token_list[i+1][1] and (token_list[i-1][0].text  in ["range","from"] or token_list[i-2][0].text  in ["range","from"] or token_list[i-3][0].text in ["range","from"])):
                for j in range(i+2,len(token_list)):
                    if token_list[j][1] == token_list[i+1][1]:
                        token_list[j][1] = token_list[i][1]
                    else:
                        break
                token_list[i+1][1] = token_list[i][1]
            if (str_is_num(tok[0].text) and i>0  and i + 2 < len(token_list) and token_list[i+1][0].text in ["and","or","but"] and tok[1] != token_list[i-1][1]) :
                token_list[i][1] = token_list[i-1][1]
            
            if tok[0].lemma_ in NEGATIVE_WORDS and token_list[i][1] == 0 :
                if token_list[i][1] != token_list[i+1][1]:
                    token_list[i][1] = token_list[i+1][1]
                elif i + 4 < len(token_list) and token_list[i][1] != token_list[i+2][1]:
                    for iii in range(2):
                        token_list[i+iii][1] = token_list[i+2][1]
                elif i + 5 < len(token_list) and token_list[i][1] != token_list[i+3][1]:
                    for iii in range(3):
                        token_list[i+iii][1] = token_list[i+3][1]
                elif i + 6 < len(token_list) and token_list[i][1] != token_list[i+4][1]:
                    for iii in range(4):
                        token_list[i+iii][1] = token_list[i+4][1]
        return token_list
    token_list = and_for_next_condition(token_list)
    token_list = repair_disjoin_phase(token_list)
    return token_list


def sentence_dump(sentence,sent_data):
    
    tokens = [token for token in sentence]
    root = [token for token in sentence if token.head == token]
    if len(root) > 1:
        return None
    root = root[0]

    sent_data['root'].append(root.i)
    for token in sentence:
        token_data = []
        for child in token.children:
            one_obj = {}
            one_obj['dep_'] = child.dep_
            one_obj['idx'] = child.i
            token_data.append(one_obj)
        sent_data['data'].append(token_data)  
    
    sent_data['data'] = merge_punctuation_2(sentence,sent_data['data'])
    return sent_data






def merge_noun_chunks(doc):
    """Merge noun chunks into a single token.
    doc (Doc): The Doc object.
    RETURNS (Doc): The Doc object with merged noun chunks.
    DOCS: https://spacy.io/api/pipeline-functions#merge_noun_chunks
    """
    if not doc.is_parsed:
        return doc
    with doc.retokenize() as retokenizer:
        for np in doc.noun_chunks:
            if '"' not in np.text and "'" not in np.text:
                attrs = {"tag": np.root.tag, "dep": np.root.dep}
                retokenizer.merge(np, attrs=attrs)
    return doc



def col_match_main(tables,ts,schema):
    def col_match(tables,ts,schema,input_match=None,only_exact_match=False):
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
            match = [[] for i in range(len(col_matchs[0]))]
            for cm in col_matchs:
                for i,c in enumerate(cm):
                    if c:
                        for d1,d2,d3,d4 in zip(c[0],c[1],c[2],c[3]):
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

    match = col_match(tables,ts,schema)
    new_tables = [i for i in range(len(schema.table_names_original)) if i not in tables]
    if match.count([]) == len(match) and -1 not in tables and new_tables:
        match = col_match(new_tables,ts,schema)
    elif -1 not in tables and new_tables:
        match = col_match(new_tables,ts,schema,match,only_exact_match=True)
    if match.count([]) == len(match) and tables:
        match = col_or_match(ts,tables,match,schema)
    return match



def sentence_cut_analyze(sentences, table_matchs, list_idxs, schema, sub_q, q_sql, token_list,run_time,patterns):
    def there_is_table_in_the_end(col_previous,tab_previous):
        there_is_table = False
        for col,tab in zip(reversed(col_previous),reversed(tab_previous)):
            if tab:
                there_is_table = True
                break
            if col:
                there_is_table = False
                break
        return there_is_table
    def col_in_tables(previous_col_match,tables,schema):
        for cols in previous_col_match:
            if cols:
                for col in cols[0]:
                    if schema.column_tokens_table_idx[col]  in tables:
                        return True
        return False
    others = []
    previous_col_match = []
    skip_once = False
    conditional_skip = False
    previous_db_match = {}
    select_tables = []
    for sentence, table_match, list_idx, type_ in zip(sentences,table_matchs,range(len(table_matchs)),sub_q.sub_sequence_type):
        if type_ == 0:
            tables = look_for_table_idx(sub_q, list_idx, 0, schema)
            if len(tables) >= 1 and tables != [-1]:
                select_tables.extend(tables)
        else:
            break
    pattern_for_return = []
    for sentence, table_match, list_idx, type_ in zip(sentences,table_matchs,range(len(table_matchs)),sub_q.sub_sequence_type):
        
        tables = look_for_table_idx(sub_q, list_idx, 0, schema)
        if len(tables) < 1 or tables == [-1]:
            tables = [i for i in range(len(schema.table_tokens))]
            db_tables = tables
        elif type_ > 0:
            db_tables = select_tables + tables
        else:
            db_tables = tables

        ts = TokenString(None,sub_q.question_tokens[list_idx])

        full_match = col_match_main(tables,ts,schema)
        previous_col_match.append(full_match)

        if skip_once:
            skip_once = False
            pr,db_match = pattern_reconize(ts,table_match,full_match,sub_q.sequence_entt[list_idx],schema,tables,[[['START', 'SEARCH', 'DATABASE']]],[None],in_db_match=sub_q.db_match[list_idx],db_table_idxs = db_tables)
            previous_db_match[list_idx-1].extend(db_match)
            pattern_for_return.append([pattern_recomand(ts,table_match,full_match,sub_q.sequence_entt[list_idx],db_match,schema,tables,False)[1],-1])
            continue

        
        tks = PATTERNS_TOKS if  sub_q.sentence_num == 2 and (sub_q.sub_sequence_type.count(0) >= 2 or (list_idx == 0 and sub_q.question_tokens[list_idx][0].lemma_ == "which")) else [[["SKIP"]] if ip == 1 else pt for ip,pt in enumerate(PATTERNS_TOKS)]
        pr,db_match = pattern_reconize(ts,table_match,full_match,sub_q.sequence_entt[list_idx],schema,tables,tks,PATTERN_FUN,in_db_match=sub_q.db_match[list_idx],db_table_idxs = db_tables)
        previous_db_match[list_idx] = db_match
        if pr[3]:
            pattern_for_return.append([pr[3],pr[0]])
        else:
            pattern_for_return.append([pattern_recomand(ts,table_match,full_match,sub_q.sequence_entt[list_idx],db_match,schema,tables,False)[1],-1])
        if pr[0] is not None:
            if conditional_skip and pr[0] != 1:
                continue
                conditional_skip = False
            if pr[0] == 0 and type_ != 0 and run_time != 2:
                for i in (sub_q.original_idx[list_idx]):
                    token_list[i][1] = pr[0]
                    sub_q.sub_sequence_type[list_idx] = pr[0]
            elif pr[0] == 1 or pr[0] == 9:
                if list_idx < len(sub_q.original_idx) - 1 and sub_q.sub_sequence_type[list_idx]> 0 and list_idx > 0 and (sub_q.sub_sequence_type[list_idx+1] > 0) and sub_q.question_tokens[list_idx][-1].text not in ["or", "and",",","but"] and sub_q.question_tokens[list_idx+1][0].lemma_ not in ["or", "and", ",", "order","sort","but"]:
                    tables = look_for_table_idx(sub_q, list_idx+1, 0, schema)
                    next_col_match = col_match_main(tables,TokenString(None,sub_q.question_tokens[list_idx+1]),schema)
                    if not table_matchs[list_idx+1][0] and (not next_col_match[0] or pr[0] == 9 or (full_match[-1] and (full_match[-1][0][0] in next_col_match[0][0] or next_col_match[0][0][0] in full_match[-1][0]))): # change next
                        if list_idx+2 >= len(table_matchs) or sub_q.sub_sequence_type[list_idx+2] != type_:
                            for i in (sub_q.original_idx[list_idx+1]):
                                token_list[i][1] = type_
                            sub_q.sub_sequence_type[list_idx+1] = type_
                            skip_once = True
                        elif not conditional_skip:
                            for i in (sub_q.original_idx[list_idx]):
                                token_list[i][1] = sub_q.sub_sequence_type[list_idx+1]
                            sub_q.sub_sequence_type[list_idx] = sub_q.sub_sequence_type[list_idx+1]
                        else:
                            conditional_skip = False
                elif conditional_skip:
                    conditional_skip = False
                elif (list_idx == len(sub_q.original_idx) - 1 or sub_q.question_tokens[list_idx][-1].text in[".", "?",",", "and", "or"] or ((sub_q.question_tokens[list_idx+1][0].lemma_ in ["or", "and", ",", "order","sort"]) and sub_q.question_tokens[list_idx][0].tag_ == "IN") ) and sub_q.sub_sequence_type[list_idx]> 0 and list_idx > 0  and ((list_idx == len(sub_q.original_idx) - 1 and sub_q.question_tokens[list_idx][0].lemma_ not in["or", "and", ","] and sub_q.question_tokens[list_idx-1][-1].lemma_ not in["or", "and", ","] ) or (sub_q.question_tokens[list_idx][0].lemma_ not in["or", "and", ",", "order","sort"] and sub_q.question_tokens[list_idx-1][-1].lemma_ not in["or", "and", ",", "order","sort"]) ):
                    follow_last = True if pr[0] == 1 else False
                    if ts.tag_.startswith("W") and list_idx >= 1:
                        all_tables = []
                        for i in range(0,list_idx):
                            for tbm in table_matchs[i]:
                                all_tables.extend(tbm)
                            for cm in previous_col_match[i]:
                                if cm:
                                    for c in cm[0]:
                                        all_tables.append(schema.column_names_original[c][0])
                        for tbm in table_match:
                            if tbm:
                                check_follow_last = [ t not in all_tables for t in tbm]
                                if len(check_follow_last) == check_follow_last.count(True):
                                    follow_last = False
                                    break
                    if follow_last:
                        for i in (sub_q.original_idx[list_idx]):
                            token_list[i][1] = sub_q.sub_sequence_type[list_idx-1]
                        sub_q.sub_sequence_type[list_idx] = sub_q.sub_sequence_type[list_idx-1]
                elif (list_idx == len(sub_q.original_idx) - 1 or (list_idx>0 and sub_q.sub_sequence_type[list_idx-1]==0)) and pr[3] in [["+","TABLE","PCOL"],["+","PCOL"],["+","PCOL","*"]]:
                    for i in (sub_q.original_idx[list_idx]):
                        token_list[i][1] = 0
                    sub_q.sub_sequence_type[list_idx] = 0
            elif (pr[0] == 2 or pr[0] == 10) and list_idx > 0:
                if pr[0] == 10 and sub_q.sub_sequence_type[list_idx-1] == 0:
                    continue
                if pr[0] == 2 and pr[3][-1] in ['COL'] and  list_idx + 1 < len(sub_q.original_idx) and sub_q.question_tokens[list_idx+1][0].text not in ['and','or',',','.','?'] and sub_q.question_tokens[list_idx][-1].text not in ['and','or',',','.','?'] and (not patterns or patterns[list_idx+1][1] != 5) and (not previous_col_match[list_idx-1][-1] or not (previous_col_match[list_idx-1][-1] == previous_col_match[list_idx][0] or (len(previous_col_match[list_idx])>1 and previous_col_match[list_idx-1][-1] == previous_col_match[list_idx][1]))):
                    if sub_q.question_tokens[list_idx+1][0].lemma_ in ABSOLUTELY_GRSM_DICT or sub_q.question_tokens[list_idx+1][0].tag_ == "IN" or  there_is_table_in_the_end(previous_col_match[list_idx-1],table_matchs[list_idx-1]):
                        target_idx = list_idx-1 if list_idx>0 and sub_q.sub_sequence_type[list_idx-1] == 0 and sub_q.question_tokens[list_idx][0].text == "in" and there_is_table_in_the_end(previous_col_match[list_idx],table_matchs[list_idx]) and col_in_tables(previous_col_match[list_idx-1],tables,schema) else list_idx+1
                        for i in (sub_q.original_idx[list_idx]):
                            token_list[i][1] = sub_q.sub_sequence_type[target_idx]
                        sub_q.sub_sequence_type[list_idx] = sub_q.sub_sequence_type[target_idx]
                        continue
                if sub_q.question_tokens[list_idx-1][-1].text not in [".", "?"] and ((sub_q.question_tokens[list_idx-1][-1].text not in ["or", "and",",","but"] and sub_q.question_tokens[list_idx][0].text not in ["or", "and",",","but"]) or ((sub_q.question_tokens[list_idx-1][-1].text in ["and",","] or sub_q.question_tokens[list_idx][0].text in ["and",","]) and sub_q.sub_sequence_type[list_idx-1] == 0)) :
                    for i in (sub_q.original_idx[list_idx]):
                        token_list[i][1] = sub_q.sub_sequence_type[list_idx-1]
                    sub_q.sub_sequence_type[list_idx] = sub_q.sub_sequence_type[list_idx-1]
            elif pr[0] == 3 and type_ == 0 and list_idx>0:
                for i in (sub_q.original_idx[list_idx]):
                    token_list[i][1] = sub_q.sub_sequence_type[list_idx-1] + 1
                sub_q.sub_sequence_type[list_idx] = sub_q.sub_sequence_type[list_idx-1] + 1
            elif pr[0] == 4 and type_ > 0:
                there_is_table = True
                if list_idx > 0 and list_idx + 1 < len(sub_q.sequence_entt) and sub_q.sub_sequence_type[list_idx-1] == 0:
                    there_is_table = False
                    for t in table_matchs[list_idx-1]:
                        if t:
                            there_is_table = True
                            break
                if list_idx > 0 and (list_idx + 1 == len(sub_q.sequence_entt) or not there_is_table or sub_q.question_tokens[list_idx][-1].text in ['?','.',',','and','or','but'] or sub_q.question_tokens[list_idx+1][0].text in ['?','.',',','and','or','but',"where","whose","which","that"]):
                    # follow previous type:
                    for i in (sub_q.original_idx[list_idx]):
                        token_list[i][1] = sub_q.sub_sequence_type[list_idx-1]
                        sub_q.sub_sequence_type[list_idx] = sub_q.sub_sequence_type[list_idx-1]
                    
                elif sub_q.sub_sequence_type[list_idx+1] > 0: # set this sub string to following next type
                    for i in (sub_q.original_idx[list_idx]):
                        token_list[i][1] = sub_q.sub_sequence_type[list_idx+1]
                        sub_q.sub_sequence_type[list_idx] = sub_q.sub_sequence_type[list_idx+1]
            elif pr[0] == 5:
                if "DATE" not in pr[3] and list_idx > 1 and sub_q.sub_sequence_type[list_idx] == sub_q.sub_sequence_type[list_idx-1]: # recover
                    for i in (sub_q.original_idx[list_idx]):
                        token_list[i][1] = sub_q.sub_sequence_type[list_idx-1]+1
                    sub_q.sub_sequence_type[list_idx] = sub_q.sub_sequence_type[list_idx-1]+1
                pass
            elif pr[0] == 6:
                if list_idx+1 < len(sub_q.sub_sequence_type) and ((sub_q.sub_sequence_type[list_idx+1] > 0 and " ' " in sub_q.sub_sequence_list[list_idx+1]) or (list_idx == 0 and len(sub_q.sub_sequence_type) > 1 and sub_q.sub_sequence_type[list_idx+1] == 0)):
                    for i in (sub_q.original_idx[list_idx]):
                        token_list[i][1] = sub_q.sub_sequence_type[list_idx+1]
                        sub_q.sub_sequence_type[list_idx] = sub_q.sub_sequence_type[list_idx+1]
                elif list_idx > 0:
                    for i in (sub_q.original_idx[list_idx]):
                        token_list[i][1] = sub_q.sub_sequence_type[list_idx-1]
                        sub_q.sub_sequence_type[list_idx] = sub_q.sub_sequence_type[list_idx-1]
            elif pr[0] == 8 and list_idx + 1 < len(sub_q.sub_sequence_list) and sub_q.sentence_num == 2 and " also " not in " ".join(sub_q.sub_sequence_list).lower() and " And " not in " ".join(sub_q.sub_sequence_list):
                tables = look_for_table_idx(sub_q, list_idx+1, 0, schema)
                next_col_match = col_match_main(tables,TokenString(None,sub_q.question_tokens[list_idx+1]),schema)
                if len(sub_q.original_idx[list_idx]) <= 3 or (not table_matchs[list_idx+1][0] and not next_col_match[0]): # change next
                    if type_ != 0:
                        for i in (sub_q.original_idx[list_idx+1]):
                            token_list[i][1] = type_
                        sub_q.sub_sequence_type[list_idx+1] = type_
                    else:
                        for i in (sub_q.original_idx[list_idx]):
                            token_list[i][1] = sub_q.sub_sequence_type[list_idx+1]
                        sub_q.sub_sequence_type[list_idx] = sub_q.sub_sequence_type[list_idx+1]
                    conditional_skip = True
    return token_list,previous_col_match,previous_db_match,pattern_for_return



def generate_sq_qsql(token_list,sql,schema,sentence_num=1):
    tokens = [i[0] for i in token_list]
    question_type = [i[1] for i in token_list]
    sq = SubQuestion(sql["question"],question_type,sql["table_match"],sql["question_tag"],sql["question_dep"],sql["question_entt"],sql,run_special_replace=False,sentence_num=sentence_num)
    if len(tokens) == len(question_type):
        qsql = QuestionSQL(sq,None,tokens)
    else:
        qsql = QuestionSQL(sq,_tokenizer)
    sq.tokenize(qsql)
    return sq,qsql

def pattern_sentence_analyse(token_list,sql,schema,sentence_num,run_time,patterns=None):
    sq,qsql = generate_sq_qsql(token_list,sql,schema,sentence_num)
    token_list,previous_col_match,previous_db_match,pattern_for_return = sentence_cut_analyze(sq.sub_sequence_list,sq.table_match, sq.original_idx, schema,sq,qsql,token_list,run_time,patterns)
    delete_list = []
    for i,t in enumerate(sq.sub_sequence_type):
        if i > 0 and t == sq.sub_sequence_type[i-1]:
            pattern_for_return[i-1][0] = copy.deepcopy(pattern_for_return[i-1][0]) + copy.deepcopy(pattern_for_return[i][0])
            delete_list.append(i)
    for i in reversed(delete_list):
        del pattern_for_return[i] 
    i = 0
    for sqts in sq.table_match:
        for sqt in sqts:
            if not sqt and sql["table_match"][i]:
                sql["table_match"][i] = []
            i += 1
    return token_list,previous_col_match,previous_db_match,pattern_for_return




def add_col_analyze(sentences, table_matchs, list_idxs, schema, sub_q, q_sql, token_list,full_col_match,full_db_match,use_pattern_generate_col):
    def add_col(sub_q,list_idx,offset,schema,col_id,token_list,p_tokens,add_word_directly=None,break_grsm=True,insert_idx = -1):
        start_idx = 0
        new_insert = []
        if insert_idx == -1:
            for p_tok,q_tok in zip(p_tokens,sub_q.question_tokens[list_idx]):
                if (break_grsm and ("GRSM" in p_tok or "JJS" in p_tok)) or "DATE" in p_tok or "YEAR" in p_tok or ("NUM" in p_tok and "DATE" not in p_tokens and "YEAR" not in p_tok) or q_tok.text == "between" or q_tok.text == "'":
                    break
                else:
                    start_idx += 1
            if start_idx == 0 or start_idx == len(p_tokens):
                start_idx = 0
                for idx in sub_q.original_idx[list_idx]:
                    if token_list[idx+offset][0].text in STOP_WORDS:
                        start_idx += 1
                    else:
                        break
        else:
            start_idx = insert_idx
        
        while start_idx >= len(sub_q.original_idx[list_idx]):
            start_idx -= 1

        if add_word_directly:
            token_list.insert(offset + sub_q.original_idx[list_idx][start_idx],[SToken(text=add_word_directly,lemma=add_word_directly,ent_type="",tag="NN"),sub_q.sub_sequence_type[list_idx]])
            new_insert.append(offset + sub_q.original_idx[list_idx][start_idx])
            offset += 1
        else:
            for tok in schema.column_tokens[col_id]:
                if tok.text == "|":
                    break
                token_list.insert(offset + sub_q.original_idx[list_idx][start_idx],[tok,sub_q.sub_sequence_type[list_idx]])
                new_insert.append(offset + sub_q.original_idx[list_idx][start_idx])
                offset += 1
        return token_list,offset,new_insert
    

    def add_sentence_or_col(qtype, pattern, sub_q,list_idx,offset,schema,col_id,token_list,p_tokens,add_word_directly=None,break_grsm=True,insert_idx = -1):
        if qtype != 0 or "SJJS" not in pattern:
            return add_col(sub_q,list_idx,offset,schema,col_id,token_list,p_tokens,add_word_directly,break_grsm,insert_idx)
        else:
            for ques in sub_q.sub_sequence_list:
                if " each " in ques:
                    if insert_idx > 3 and token_list[insert_idx-2][0].text == "of":
                        token_list[insert_idx-2][0] = SToken(text="and",tag="CC")
                    elif insert_idx > 3 and token_list[insert_idx-3][0].text == "of":
                        token_list[insert_idx-3][0] = SToken(text="and",tag="CC")
                    return add_col(sub_q,list_idx,offset,schema,col_id,token_list,p_tokens,add_word_directly,break_grsm,insert_idx)
            extract_list = []
            extract_list_idx = []
            cut = False
            for i in range(1,5):
                if insert_idx > i and sub_q.question_tokens[list_idx][insert_idx-i].text in ["of","for"]:
                    cut = True
                    break
                else:
                    extract_list.append(sub_q.question_tokens[list_idx][insert_idx-i])
                    extract_list_idx.append(insert_idx-i)
            if not cut and ( (insert_idx+1 < len(sub_q.question_tokens[list_idx]) and sub_q.question_tokens[list_idx][insert_idx+1].text == "'s") or  (insert_idx > 3 and (str_is_num(sub_q.question_tokens[list_idx][insert_idx-2].text) or str_is_num(sub_q.question_tokens[list_idx][insert_idx-3].text))) or sub_q.question_tokens[list_idx][0].lower_ in ['who','where'] or (insert_idx+1 < len(sub_q.question_tokens[list_idx]) and insert_idx+3 >= len(sub_q.question_tokens[list_idx]) and len(sub_q.question_tokens) == 1 and p_tokens[insert_idx+1] in ["COL","TABLE-COL"]) ):
                cut = True
                extract_list = []
                extract_list_idx = []
                for i in range(1,5):
                    if insert_idx > i and sub_q.question_tokens[list_idx][insert_idx-i].text in ["of","for",'the','is','are','be','was','were']:
                        cut = True
                        break
                    else:
                        extract_list.append(sub_q.question_tokens[list_idx][insert_idx-i])
                        extract_list_idx.append(insert_idx-i)
            if cut:
                if list_idx + 1 < len(sub_q.sub_sequence_type):
                    token_list_type = sub_q.sub_sequence_type[list_idx+1] + 1
                else:
                    token_list_type = sub_q.sub_sequence_type[list_idx] + 1

                if sub_q.question_tokens[list_idx][-1].text in [',','.','?']:
                    last_idx = -2
                else:
                    last_idx = -1
                
                offset += 1
                token_list.insert(offset + sub_q.original_idx[list_idx][last_idx],[SToken(text="that",ent_type="",tag="WP"),token_list_type])
                new_insert.append(offset + sub_q.original_idx[list_idx][last_idx])
                offset += 1
                token_list.insert(offset + sub_q.original_idx[list_idx][last_idx],[SToken(text="has",ent_type="",tag="VP"),token_list_type])
                new_insert.append(offset + sub_q.original_idx[list_idx][last_idx])
                offset += 1
                for tok in reversed(extract_list):
                    token_list.insert(offset + sub_q.original_idx[list_idx][last_idx],[tok,token_list_type])
                    new_insert.append(offset + sub_q.original_idx[list_idx][last_idx])
                    offset += 1
                for i in reversed(extract_list_idx):
                    token_list[i] = [SToken(text="*"+token_list[i][0].text+'*',ent_type="",tag=""),token_list[i][1]]

                if add_word_directly:
                    token_list.insert(offset + sub_q.original_idx[list_idx][last_idx],[SToken(text=add_word_directly,lemma=add_word_directly,ent_type="",tag="NN"),token_list_type])
                    new_insert.append(offset + sub_q.original_idx[list_idx][last_idx])
                    offset += 1
                else:
                    for tok in schema.column_tokens[col_id]:
                        if tok.text == "|":
                            break
                        token_list.insert(offset + sub_q.original_idx[list_idx][last_idx],[tok,token_list_type])
                        new_insert.append(offset + sub_q.original_idx[list_idx][last_idx])
                        offset += 1
                offset -= 1
                return  token_list,offset,new_insert
        return  token_list,offset,[]


    def there_is_time_type(full_match,schema,time_type='time'):
        count = set()
        for cols in full_match:
            if cols:
                for c in cols[0]:
                    if schema.column_types[c] == time_type or schema.column_tokens_lemma_str[c] in ["date","year"]:
                        count.add(c)
        return len(count)

    def add_full_time_type_col(full_match,schema,sub_q,list_idx,offset,token_list,pattern_guess):
        for i,cols in enumerate(full_match):
            if cols:
                for j,c in enumerate(cols[0]):
                    if cols[1][j] == 1 and cols[2][j] == 0 and schema.column_types[c] == 'time':
                        if schema.column_tokens_lemma_str[c].count(" ") > 0:
                            for tok in schema.column_tokens_lemma_str[c].split(" "):
                                if tok != sub_q.question_tokens[list_idx][i].lemma_ and tok != sub_q.question_tokens[list_idx][i].text:
                                    return add_col(sub_q,list_idx,offset,schema,None,token_list,pattern_guess,add_word_directly=tok,insert_idx = i)
        return None,None,None

    others = []
    previous_col_match = []
    new_insert_idxs = []
    offset = 0
    for sentence, table_match, list_idx, type_ in zip(sentences,table_matchs,range(len(table_matchs)),sub_q.sub_sequence_type):
        new_insert = []
        tables = look_for_table_idx(sub_q, list_idx, 0, schema)
        if len(tables) < 1 or tables == [-1]:
            tables = [i for i in range(len(schema.table_tokens))]
        
        ts = TokenString(None,sub_q.question_tokens[list_idx])
        full_match = col_match_main(tables,ts,schema)
        previous_col_match.append(full_match)
        from .pattern_question_type import ADD_COL_PATTERNS as PATTERNS_TOKS,ADD_COL_PATTERN_FUN as PATTERN_FUN
        pr,db_match = pattern_reconize(ts,table_match,full_match,sub_q.sequence_entt[list_idx],schema,tables,PATTERNS_TOKS,PATTERN_FUN)
        if list_idx in full_db_match.keys() and len(full_db_match[list_idx]) == len(db_match) :
            db_match = full_db_match[list_idx]
        if pr[0] is not None:
            if pr[0] != 6 and not use_pattern_generate_col:
                continue
            if pr[0] == 1: 
                pattern_guess = pattern_recomand(ts,table_match,full_match,sub_q.sequence_entt[list_idx],db_match,schema,tables,False)
                # There is SGRSM and SJJS means that we can find COL from schema for the SGRSM and SJJS:
                # So we need to check the COL is correct or not:
                do_not_need_add = False
                p_toks = pattern_guess[2].split(" | ")

                if (" old " not in ts.lemma_ and " young " not in ts.lemma_) and ((("GRSM than COL" in pattern_guess[0] and not ('than,(than) | AGG,COL' in pattern_guess[2] and "SGRSM than COL" in pattern_guess[0])) or "JJS COL" in pattern_guess[0] or "DB" in pattern_guess[0])):
                    do_not_need_add = True
                elif "SJJS COL" in pattern_guess[0] and (" old id " in ts.lemma_ or " young id " in ts.lemma_ ):
                    do_not_need_add = True
                elif "(high)" in pattern_guess[2] and "COL" in pattern_guess[0]:
                    do_not_need_add = True
                elif "SGRSM" not in pattern_guess[0] and "SJJS" not in pattern_guess[0] and " most " not in ts.text and " more " not in ts.text:
                    do_not_need_add = True
                elif "COL" in pattern_guess[0]:
                    for i, pt in enumerate(p_toks):
                        if "SGRSM" in pt or "SJJS" in pt:
                            last_word = pt.split(",")[-1]
                            if last_word[1:-1] in ["late","young","old","early","elderly","new","quick","fast","slow","long"]:
                                for fcol in full_match:
                                    if fcol:
                                        for c in fcol[0]:
                                            if schema.column_types[c] in ["year",'time']:
                                                do_not_need_add = True
                                                break
                                    if do_not_need_add:
                                        break
                            if last_word[1:-1] in S_ADJ_WORD_DIRECTION.keys():
                                n_words = [w_tok[0] for w_tok in S_ADJ_WORD_DIRECTION[last_word[1:-1]]]
                                for w_tok,k in zip(reversed(ts.tokens),range(len(ts.tokens)-1,-1,-1)):
                                    if w_tok.lemma_ in n_words:
                                        do_not_need_add = True
                                        break
                                    elif w_tok.tag_ == "IN" and k < i:
                                        break
                                if do_not_need_add:
                                    break
                                time_count = there_is_time_type(full_match,schema)
                                if "date" in n_words and "year" in n_words and time_count:
                                    if time_count == 1:
                                        tmp = add_full_time_type_col(full_match,schema,sub_q,list_idx,offset,token_list,pattern_guess[1])
                                        if tmp != (None,None,None) and type_ != 0:
                                            token_list,offset,new_insert = tmp
                                    do_not_need_add = True
                                    break
                if not do_not_need_add:
                    # add new col:
                    table_idxs = tables if tables else [-1]
                    for i, pt in enumerate(p_toks):
                        if "SGRSM" in pt or "SJJS" in pt:
                            last_word = pt.split(",")[-1]
                            if last_word[1:-1] not in S_ADJ_WORD_DIRECTION.keys():
                                last_word = "(" + lstem.stem(last_word[1:-1]) + ")"
                            if last_word[1:-1] in S_ADJ_WORD_DIRECTION.keys():
                                agg_id, col_id, sgrsm_word = get_AWD_column(last_word[1:-1], table_idxs, schema, True, all_tokens=token_list)
                                if ((len(col_id) == 1 and col_id[0] < 0) or len(col_id) != 1) and not sgrsm_word and i+1<len(table_match) and table_match[i+1] :
                                    agg_id, col_id, sgrsm_word = get_AWD_column(last_word[1:-1], table_match[i+1], schema, True, all_tokens=token_list)
                                if len(col_id) == 1 and col_id[0] > 0:
                                    token_list,offset,new_insert = add_sentence_or_col(type_,pt, sub_q,list_idx,offset,schema,col_id[0],token_list,pattern_guess[1], insert_idx = -1 if type_ != 0 and "1&SJJS" not in pr[3] else i+1 )
                                    break
                                elif sgrsm_word:
                                    token_list,offset,new_insert = add_sentence_or_col(type_,pt, sub_q,list_idx,offset,schema,col_id[0],token_list,pattern_guess[1],sgrsm_word, insert_idx = -1 if type_ != 0 and "1&SJJS" not in pr[3] else i+1 )
                                    break
            elif pr[0] == 2 and (type_ != 0 or " between 1" in ts.text or " between 2" in ts.text): # There is DATE without COL
                pattern_guess = pattern_recomand(ts,table_match,full_match,sub_q.sequence_entt[list_idx],db_match,schema,tables,False)
                should_add = True
                if ("between" in pattern_guess[0] and not ("between DATE DATE" in pattern_guess[0] or "between YEAR YEAR" in pattern_guess[0] or "between DATE and DATE" in pattern_guess[0] or "between YEAR and YEAR" in pattern_guess[0])):
                    should_add = False
                elif "COL " in pattern_guess[0] or " COL" in pattern_guess[0]:
                    if "YEAR" in pattern_guess[0]:
                        should_add = not there_is_time_type(full_match,schema,'year')
                    else:
                        should_add = not there_is_time_type(full_match,schema)
                if should_add:
                    table_idxs = tables if tables else [-1]
                    date_types = set()
                    for pgs, tok_tmp, tok_idx in zip(pattern_guess[1],ts.tokens,range(len(ts.tokens))):
                        if pgs in ["DATE","YEAR"]:
                            dt = str_is_date(tok_tmp.text,ts.tokens,tok_idx)
                            if dt:
                                date_types.add(dt)
                    if len(date_types) == 1:
                        date_types = list(date_types)[0]
                    else:
                        date_types = "DATE"
                    agg_id, col_id, word = get_col_from_related_word(date_types, table_idxs, schema, True)
                    if word == "year" and "YEAR" in pattern_guess[1]:
                        y_idx = pattern_guess[1].index("YEAR")
                        if y_idx > 0 and pattern_guess[1][y_idx-1] == "between":
                            y_idx -= 1
                        token_list,offset,new_insert = add_col(sub_q,list_idx,offset,schema,col_id[0],token_list,pattern_guess[1],word,break_grsm=False,insert_idx = y_idx )
                    elif len(col_id) == 1 and col_id[0] > 0:
                        token_list,offset,new_insert = add_col(sub_q,list_idx,offset,schema,col_id[0],token_list,pattern_guess[1],break_grsm=False)
                    elif word:
                        token_list,offset,new_insert = add_col(sub_q,list_idx,offset,schema,col_id[0],token_list,pattern_guess[1],word,break_grsm=False)
            elif pr[0] == 3 and type_ != 0:
                pattern_guess = pattern_recomand(ts,table_match,full_match,sub_q.sequence_entt[list_idx],db_match,schema,tables,False)
                if "COL " not in pattern_guess[0] and " COL" not in pattern_guess[0] and "TABLE " not in pattern_guess[0] and " TABLE" not in pattern_guess[0]:
                    print("please add SGRSM!!!!!!!!!!!!!!!!!!!!!!!!!")
            elif pr[0] == 5:
                  print("??unrecognized each : "+sub_q.sub_sequence_list[list_idx])
            elif pr[0] == 6:
                old = sub_q.sub_sequence_type[list_idx]
                sub_q.sub_sequence_type[list_idx] = -1
                if 0 in sub_q.sub_sequence_type or old != 0:
                    sub_q.sub_sequence_type[list_idx] = old
                    for ei in range(sub_q.original_idx[list_idx][0],sub_q.original_idx[list_idx][-1]+2):
                        if ei >= len(token_list):
                            break
                        if token_list[ei][1] != sub_q.sub_sequence_type[list_idx]:
                            if ei > sub_q.original_idx[list_idx][-1]:
                                break
                        else:
                            token_list[ei][1] = -1
                else:
                    sub_q.sub_sequence_type[list_idx] = 0
            elif pr[0] == 7 and list_idx >= 1:
                need_add = True
                if sub_q.col_match[list_idx].count([]) == len(sub_q.col_match[list_idx]):
                    for tokk in sentence.split(" "):
                        if tokk.isdigit() and int(tokk) > 99:
                            need_add = False
                    if need_add:
                        need_add = False
                        for ti in range(len(sub_q.table_match[list_idx-1])-1,-1,-1):
                            if sub_q.table_match[list_idx-1][ti]:
                                for tt in sub_q.table_match[list_idx-1][ti]:
                                    if "age" in schema.table_col_lemma[tt]:
                                        need_add = True
                                break
                        if need_add:
                            token_list,offset,new_insert = add_col(sub_q,list_idx,offset,schema,-2,token_list,[],"age",break_grsm=False,insert_idx = 0 )
        new_insert_idxs.extend(new_insert)
    if new_insert_idxs:
        check_new_insert_idxs = copy.deepcopy(new_insert_idxs)
        check_new_insert_idxs.sort()
        assert check_new_insert_idxs == new_insert_idxs
        check_new_insert_idxs = set(check_new_insert_idxs)
        assert len(check_new_insert_idxs) == len(new_insert_idxs)
    return token_list,offset,new_insert_idxs


def pattern_generate_col(token_list,sql,schema,previous_col_match,previous_db_match,use_pattern_generate_col,sentence_num):
    sq,qsql = generate_sq_qsql(token_list,sql,schema,sentence_num)
    return add_col_analyze(sq.sub_sequence_list,sq.table_match, sq.original_idx, schema,sq,qsql,token_list,previous_col_match,previous_db_match,use_pattern_generate_col)





def easy_cut(token_list,table_match,col_match):
    all_zero = True
    for tok in token_list:
        if tok[1] != 0:
            all_zero = False
            break
    easy_cut_pattern = [
        [0,"in","which"],
        [0,"whose"],
        [0,"which"],
        [1,",","sort"],
        [1,",", "and" ,"sort"],
        [1, "and" ,"sort"],
        [1,",", "and" ,"order"],
        [1, "and" ,"order"],
        [1,"sort", "in" ,"the"],
        [1,"sort", "by"],
        [0,"by","*","*","in","*","order"],
        [0,"by","*","in","*","order"],
        [0,"by","*","order"],
        [0,"by","*","*","order"],
        [0,"in","*","order"],
        [0,"in","*","*", "order"],
        [0,"ordered","*"],
        [0,"*","*","*", "who"],
        [0,"order","by","*"],        
    ]

    for pattern in easy_cut_pattern:
        if not pattern[0] and not all_zero:
            continue
        pattern = pattern[1:]
        for i, tok in enumerate(token_list):
            if i + len(pattern) >= len(token_list):
                break
            match = True if tok[1] == 0 else False
            for j, p in enumerate(pattern):
                if p == "*":
                    continue
                elif (token_list[i+j][0].text != p and token_list[i+j][0].lemma_ != p) or  table_match[i+j] or col_match[i+j]:
                    match = False
                    break
            if match:
                offset = 0
                for j, p in enumerate(pattern):
                    if p != "*":
                        break
                    offset += 1
                if all_zero and len(table_match) == len(col_match):
                    there_is_col_or_table = False
                    for j in range(0,i+offset):
                        if table_match[j] or col_match[j]:
                            there_is_col_or_table = True
                            break
                    if not there_is_col_or_table:
                        continue
                next_type = 0
                stop_idx = len(token_list)
                for z in range(i+offset,len(token_list),1):
                    if token_list[z][1] != next_type:
                        next_type = token_list[z][1]
                        stop_idx = z
                        break
                if next_type == 0:
                    next_type = 1
                for z in range(i+offset,stop_idx,1):
                    if (token_list[z][0].text in [",","and",".","?"] and z-i < len(pattern) and pattern[z-i] not in [",","and",".","?"])  and z + 3 < len(token_list): 
                        break
                    token_list[z][1] = 1
                return token_list
    return token_list


def two_setence_analyse(sents,token_list):
    if len(sents) == 2:
        if sents[0].lemma_.startswith("who be "):
            for i,tok in enumerate(token_list):
                if tok[1] != 0:
                    return token_list
                token_list[i][1] = 1
                if tok[0].lower_ in [".","?"]:
                    break
    return token_list


def select_split(token_list,col_match_list,db_match):
    def how_many_or_how_much():
        if i >= 2 and token_list[i-2][0].lemma_ == "how" and token_list[i-1][0].lemma_ in ["much","many"] and i + 1 < len(token_list) and col_match_list[i] and col_match_list[i+1]:
            col_m_l = col_match_list[i]
            col_m_r = col_match_list[i+1]
            for cm in col_m_l[0]:
                if cm in col_m_r:
                    return False
            return True
        if i >= 3 and token_list[i-3][0].lemma_ == "how" and token_list[i-2][0].lemma_ in ["much","many"] and i + 1 < len(token_list) and col_match_list[i] and col_match_list[i+1]:
            col_m_l = col_match_list[i]
            col_m_r = col_match_list[i+1]
            for cm in col_m_l[0]:
                if cm in col_m_r[0]:
                    return False
            return True
    def split(idx):
        max_ = -1
        for tl in token_list:
            if tl[1] > max_:
                max_ = tl[1]
        max_ += 1
        for ii in range(idx,len(token_list)):
            token_list[ii][1] = max_
        return token_list
    
    def there_are_condition_values():
        type_start = token_list[i][1]
        for j in range(i+1,len(token_list)):
            if token_list[j][1] != token_list[j-1][1]:
                break
            if  str_is_num(token_list[j][0].lemma_) or db_match_list[j]:
                return True
        return False

    db_match_list = []
    for dm in db_match:
        db_match_list.extend(db_match[dm])
    if len(col_match_list) != len(db_match_list) or len(db_match_list) != len(token_list):
        return
    for i,tl in enumerate(token_list):
        if how_many_or_how_much() and there_are_condition_values() and token_list[i][1] == token_list[i+1][1]:
            token_list = split(i+1)
            return token_list
        if tl[0].lemma_ == "neither"  and there_are_condition_values() and token_list[i][1] == token_list[i+1][1]:
            token_list = split(i)
            return token_list

def reset_uncontinue_type(token_list,sentence_num):
    last_tl = token_list[0][1]
    tl_type_set = set()
    tl_type_set.add(last_tl)

    if token_list[0][1] > 0 and token_list[1][1] == 0 and token_list[2][1] == 0:
        token_list[0][1] = 0

    for i,tl in enumerate(token_list):
        tl_type = tl[1]
        if tl_type != last_tl:
            if i + 1 == len(token_list):
                token_list[i][1] = last_tl
                continue
            count = 0
            for j in range(i+1,len(token_list),1):
                if token_list[j][1] == tl_type:
                    if token_list[j][0].text not in ["?",".",","]:
                        count += 1
                else:
                    break
            if (tl_type not in tl_type_set or (tl_type>2 and (token_list[i][0].text in [",","but","and","or",".","?"] or token_list[i-1][0].text in [",","but","and","or",".","?"])) or last_tl == 0) and count != 0: # means it will have at least two token for this type
                last_tl = tl_type
                tl_type_set.add(last_tl)
            else:
                if count <= 2:
                    if i+count+1 < len(token_list) and tl[0].lemma_ in SELECT_FIRST_WORD and tl_type == 0 and not tl[0].text.islower():
                        for j in range(i+count+2,len(token_list),1):
                            if token_list[j][1] == token_list[j-1][1]:
                                token_list[j][1] = token_list[i][1]
                        token_list[i+count+1][1] = token_list[i][1]
                        last_tl = token_list[j][1]
                    elif count >= 1 and token_list[i][0].text in [",","but","and","or",".","?"] or token_list[i-1][0].text in [",","but","and","or",".","?"] and i + count + 1 <len(token_list):
                        for j in range(i,i+count+1,1):
                            token_list[j][1] = token_list[i+count+1][1]
                        last_tl = token_list[i+count+1][1]
                        tl_type_set.add(last_tl)
                    else:
                        for j in range(i,i+count+1,1):
                            token_list[j][1] = last_tl
                else:
                    last_tl = tl_type
    return token_list



def special_word_modify(question,schema):
    question_toks = [SToken(text=tok) for tok in question.split(" ")]
    if question[-1] == "." and question_toks[-1].text != '.':
        question = question[:-1] + " ."
        question_toks = [SToken(text=tok) for tok in question.split(" ")]
    elif question[-1] == "?" and question_toks[-1].text != '?':
        question = question[:-1] + " ?"
        question_toks = [SToken(text=tok) for tok in question.split(" ")]
    elif len(question_toks[-1].text) > 1:
        question_toks.append(SToken(text='.'))
    
    if len(question_toks[-1].text) == 2 and question_toks[-1].text[0].isalpha() and not question_toks[-1].text[1].isalpha():
        tmp_text = question_toks[-1].text
        question_toks[-1] = SToken(text=tmp_text[0])
        question_toks.append(SToken(text=tmp_text[1]))

    for start_idx,tok in enumerate(question_toks):
        if len(tok.text) <= 2 and not tok.text.isupper():
            continue
        elif "_" in tok.lower_:
            if datebase_match_tables(schema,tok,0,[tok],[0],True):
                continue
            question_toks[start_idx] = question_toks[start_idx].replace("_"," ")
        elif start_idx > 2 and tok.text == "what" and question_toks[start_idx-1].text == "and":
            if question_toks[start_idx-3].lower_ == "which":
                question_toks[start_idx] = SToken(text="")
            elif start_idx > 3 and question_toks[start_idx-4].lower_ == "which":
                question_toks[start_idx] = SToken(text="")
        elif tok.lemma_ == "least" and question_toks[start_idx-1].text in ["of","at"] and start_idx + 1 < len(question_toks) and question_toks[start_idx + 1].text != 'one' and    (str_is_num(question_toks[start_idx+1].text) or (start_idx+2 < len(question_toks) and str_is_num(question_toks[start_idx+2].text)) or (start_idx+3 < len(question_toks) and str_is_num(question_toks[start_idx+3].text))):
            question_toks[start_idx-1]   = SToken(text="shinier")
            question_toks[start_idx]   = SToken(text="than")
        elif (tok.lemma_ == "most" and question_toks[start_idx-1].lemma_ in ["of","at","have"] and (str_is_num(question_toks[start_idx+1].text) or (start_idx+2 < len(question_toks) and str_is_num(question_toks[start_idx+2].text)) or (start_idx+3 < len(question_toks) and str_is_num(question_toks[start_idx+3].text)))) or (tok.lemma_ == "more" and question_toks[start_idx-1].lemma_ == "no" and start_idx+2 < len(question_toks) and str_is_num(question_toks[start_idx+2].text)): 
            question_toks[start_idx-1]   = SToken(text="uglier")
            question_toks[start_idx]   = SToken(text="than")
        elif tok.lemma_ in ["reach","reached","reaching"] and str_is_num(question_toks[start_idx+1].text):
            question_toks[start_idx]   = SToken(text="shinier")
        elif tok.text in ["title","titles","titled"] and "title" not in schema.table_col_lemma[-1] and "titled" not in schema.table_col_lemma[-1] and "titles" not in schema.table_col_lemma[-1]:
            if tok.text == "title":
                question_toks[start_idx]   = SToken(text="name")
            elif tok.text == "titles":
                question_toks[start_idx]   = SToken(text="names",lemma="name")
            else:
                question_toks[start_idx]   = SToken(text="named",lemma="name")
        elif (tok.lemma_ in SYNONYM or tok.text in SYNONYM ) and tok.lemma_ not in schema.table_tokens_lemma_str and tok.text not in schema.table_tokens_text_str and tok.text not in schema.table_col_lemma[-1] and tok.lemma_ not in schema.table_col_lemma[-1] and tok.lemma_ not in schema.table_col_text[-1]:
            key_s = tok.lemma_ if tok.lemma_ in SYNONYM else tok.text
            for syn in SYNONYM[key_s]:
                if syn in schema.table_tokens_lemma_str or syn in schema.table_tokens_text_str or syn in schema.table_col_lemma[-1] or syn in schema.table_col_text[-1]:
                    question_toks[start_idx]   = SToken(text=syn)
                    break
        elif start_idx + 4 < len(question_toks) and start_idx >= 1 :
            if tok.lemma_ in ["range","from","between"] and start_idx + 4 < len(question_toks) and question_toks[start_idx+2].text != "or" and question_toks[start_idx+3].text != "or":
                if str_is_num(question_toks[start_idx+1].text) and str_is_num(question_toks[start_idx+3].text):
                    question_toks[start_idx]   = SToken(text="between")
                    question_toks[start_idx+2] = SToken(text="and")
                elif str_is_num(question_toks[start_idx+2].text) and str_is_num(question_toks[start_idx+4].text):
                    question_toks[start_idx]   = SToken(text="between")
                    question_toks[start_idx+3] = SToken(text="and")
                    question_toks[start_idx+1] = SToken(text="")
            elif tok.lemma_ == "full" and question_toks[start_idx+1].lemma_ in ["name","names"]:
                there_is_full_name = False
                for ctls in schema.column_tokens_lemma_str:
                    if "full"  in ctls and "name" in ctls:
                        there_is_full_name = True
                        break
                if not there_is_full_name:
                    question_toks[start_idx]   = SToken(text="first")
                    question_toks.insert(start_idx+2,SToken(text="name"))
                    question_toks.insert(start_idx+2,SToken(text="last"))
                    question_toks.insert(start_idx+2,SToken(text=","))
        

    
    question = " ".join([tok.text for tok in question_toks])
    return question


def anaylse_punctuate(token_list,question,schema,all_word,keep_original_question):
    def reset_next_type(token_list,i):
        if i+1 < len(token_list):
            for j in range(i+1,len(token_list),1):
                if token_list[j][1] != token_list[i][1]:
                    break
                else:
                    token_list[j][1] = token_list[i-1][1]
            if i >= 3 and j + 4 < len(token_list) and token_list[i-1][1] == token_list[j+1][1] and token_list[i-1][0].text not in ["(","'",")"]  and token_list[i-2][0].text not in ["(","'",")"]  and token_list[i-3][0].text not in ["(","'",")"] and token_list[i][0].text not in ["(","'",")"]:
                return comma(token_list,i-1,token_list[i][1],nt=token_list[j][1]+1,nj = j+1)
        token_list[i][1] = token_list[i-1][1]
        return token_list
    def jjr_check(token_list,i,tokens):
        than_type = -1
        than_idx = 0
        for j in range(i+1,len(tokens)):
            if tokens[j].tag_ in ["JJR","RBR"]:
                break
            elif tokens[j].text == "than":
                than_type = token_list[j][1]
                than_idx = j
                break
        if than_idx and than_type != token_list[i][1]:
            for j in range(i+1,than_idx+1):
                token_list[j][1] = token_list[i][1]
            for j in range(than_idx+1,len(tokens)):
                if token_list[j][1] == than_type:
                    token_list[j][1] = token_list[i][1]
                else:
                    break
        elif than_idx and than_type == 0:
            for j in range(i+1,len(token_list)):
                if token_list[j][1] != 0:
                    break
                token_list[j][1] = 9
            for j in range(i,-1,-1):
                if token_list[j][0].tag_.startswith("V") or token_list[j][0].tag_ == "IN" or token_list[j][0].text in ["that","with","which"]:
                    token_list[j][1] = 9
                    break
                token_list[j][1] = 9

        return token_list
    
    def between_set(token_list,i,t1,t2):
        for j in range(i+1,len(token_list)):
            if token_list[j][1] == token_list[i][1]:
                continue
            elif token_list[j][1] == t2:
                token_list[j][1] = t1
            else:
                break
        return token_list

    def comma(token_list,i,ot,nt=0,nj = 0):
        if not nt and not nj:
            for j in range(i+1,len(token_list)):
                if token_list[j][1] != ot:
                    nt = token_list[j][1]
                    nj = j
                    break
        if nt:
            for j in range(i,nj):
                token_list[j][1] = nt
        return token_list     


    tokens = [tok[0] for tok in token_list]
    for i,tok in enumerate(token_list):
        if i > 0 and token_list[i][1] != token_list[i-1][1]:
            if " ' " in question and get_punctuation_word(tokens,i):
                token_list = reset_next_type(token_list,i)
                continue
            elif " ( " in question and get_punctuation_word(tokens,i,punct=["(",")"]):
                token_list = reset_next_type(token_list,i)
                continue
            res = datebase_match_tables(schema,tokens[i-1],i-1,tokens,[0],True)
            if res and res[0][0][1][1] >= i:
                token_list = reset_next_type(token_list,i)
        if tokens[i].tag_ in ["JJR","RBR"]:
            token_list = jjr_check(token_list,i,tokens)
        elif tokens[i].text == "between" and i + 4 < len(tokens) and tokens[i+1].text.isdigit() and tokens[i+2].text == "and" and tokens[i+3].text.isdigit() and token_list[i][1] != token_list[i+3][1]:
            token_list = between_set(token_list,i,token_list[i][1], token_list[i+3][1])
        elif tokens[i].text == "," and i > 0 and token_list[i-1][1] > 0 and i + 2 < len(tokens) and token_list[i-1][1] == token_list[i+1][1]:
            if " ' " not in question or not get_punctuation_word(tokens,i):
                token_list = comma(token_list,i,token_list[i+1][1])
        elif tokens[i].text == "'" and i > 0 and token_list[i-1][1] != token_list[i][1] and get_punctuation_word(tokens,i+1):
            token_list = reset_next_type(token_list,i)
        elif  tokens[i].text == "or" and i + 5 < len(token_list) and i >0 and token_list[i+1][1] == token_list[i-1][1] and  token_list[i][1] == token_list[i-1][1] and tokens[i+2].text not in [",",".","?"]  and tokens[i+3].text not in [",",".","?"] and token_list[i+1][1] != token_list[i+3][1] and tokens[i+1].text.islower() and len(tokens[i+1].text) > 2 and tokens[i+1].lemma_ in all_word:
            if " ' " not in question or not get_punctuation_word(tokens,i):
                token_list = comma(token_list,i,token_list[i+1][1])
        elif  tokens[i].text in ["with","in"] and i + 2 < len(token_list) and i >0 and token_list[i-1][1] and token_list[i][1] and token_list[i][1] != token_list[i-1][1] and  tokens[i-1].lemma_ in ["end","start"]:
            if " ' " not in question or not get_punctuation_word(tokens,i):
                token_list = reset_next_type(token_list,i)
        elif  tokens[i].text == "and" and "between" not in question and i + 4 < len(token_list) and token_list[i][1] == token_list[i+1][1] and (token_list[i+1][1] != token_list[i+2][1] or token_list[i+3][1] != token_list[i+1][1]) and token_list[i][1] > 0:
            if " ' " not in question or not get_punctuation_word(tokens,i):
                token_list = comma(token_list,i+1,token_list[i+1][1])
        elif  tokens[i].text == "for" and i > 0 and token_list[i][1] == token_list[i-1][1] and i + 5 >= len(token_list) and i + 4 <= len(token_list) and tokens[i+1].text == "each":
            if " ' " not in question or not get_punctuation_word(tokens,i):
                token_list = comma(token_list,i,token_list[i][1],nt=token_list[i][1]+10,nj = len(token_list))
        elif not keep_original_question and tokens[i].text == "least" and tokens[i-1].tag_.startswith("V") and str_is_num(tokens[i+1].text) and tokens[i+1].text != "one":
            token_list[i][0] = SToken(text="shinier",tag="JJR")
        elif tokens[i].text in ['?','.'] and i + 1 < len(tokens) and i > 0 and token_list[i][1] != token_list[i-1][1] and token_list[i][1] == token_list[i+1][1]:
            token_list[i][1] = token_list[i-1][1]
        elif  tokens[i].text == "(" and token_list[i][1] != token_list[i-1][1] and get_punctuation_word(tokens,i+1,punct=["(",")"]):
            if " ' " not in question or not get_punctuation_word(tokens,i):
                for k in range(i,len(tokens)):
                    token_list[k][1] = token_list[i-1][1]
                    if tokens[k].text == ")":
                        break
        elif  (tokens[i].lemma_ in ["order","relate"]) and i + 3 < len(token_list) and token_list[i][1] != token_list[i+1][1] and (tokens[i+1].text in ["by","to"] ):
            if " ' " not in question or not get_punctuation_word(tokens,i):
                token_list = comma(token_list,i,token_list[i][1])
        elif i > 2 and  tokens[i].text in ["most","least"] and token_list[i-2][1] == token_list[i][1] and token_list[i-2][1] == 0 and tokens[i-2].text not in ["that","who","which","when"] and tokens[i-1].lemma_ == "have":
            token_list = comma(token_list,i-1,0)
        elif i > 2 and  tokens[i].text in ["most","least"] and token_list[i-3][1] == token_list[i][1] and token_list[i-3][1] == 0 and tokens[i-3].text not in ["that","who","which","when"] and tokens[i-2].lemma_ == "have" and tokens[i-1].lemma_ == "the":
            token_list = comma(token_list,i-2,0)
        elif  tokens[i].text == "to"  and i >0 and token_list[i][1] != token_list[i-1][1] and  tokens[i-1].tag_ in ["JJS","RBS"] and i+4 >= len(tokens):
            if " ' " not in question or not get_punctuation_word(tokens,i):
                token_list = reset_next_type(token_list,i)
        elif i == 1 and  tokens[i-1].lower_ == "which" and token_list[i][1] != token_list[i-1][1] and (token_list[i][1] != token_list[i+1][1] or token_list[i][1] != token_list[i+2][1]):
            if token_list[0][1] == 0:
                token_list = reset_next_type(token_list,1)
            else:
                token_list[0][1] = 0

    if len(token_list) >= 3 and token_list[-3][0].text == "in" and token_list[-3][1] == token_list[-4][1] and token_list[-2][0].text.isdigit():
        year = int(token_list[-2][0].text)
        if year > 1800 and year < 2100:
            token_list[-3][1] += 1
            token_list[-2][1] += 1
            token_list[-1][1] += 1

    return token_list


def dump_language_feature(sentences):
    sent_data = {}
    sent_data['root'] = []
    sent_data['data'] = []
    for sent in sentences.sents:
        sent_data = sentence_dump(sent,sent_data)
    return sent_data



def db_correction(token_list, col_match, sql, schema):
    def switch_tok_match(word,target):
        len_w = len(word)
        for i in range(len_w-1):
            nw = word[0:i] + word[i+1] + word[i] + word[i+2:]
            if nw in target:
                return nw
        return None
    def correct_multiple_tokens(schema,sql,start_idx,question_toks,token_list):
        db_str = get_database_string(schema, sql['db_match'][start_idx][0][0], question_toks[start_idx:start_idx + sql['db_match'][start_idx][0][1][1] - sql['db_match'][start_idx][0][1][0] + 1])
        if db_str.count(" ") < sql['db_match'][start_idx][0][1][1] - sql['db_match'][start_idx][0][1][0]:
            len_start = 0
            len_end = db_str.count(" ")
            for z in range(start_idx + sql['db_match'][start_idx][0][1][1] - sql['db_match'][start_idx][0][1][0], start_idx+len_end,-1):
                del token_list[z]
                del sql['db_match'][z]
                del sql['question_tag'][z]
                del sql['question_entt'][z]
                del sql['table_match'][z]
                del sql['question_dep']["data"][z]
                del col_match[z]
                del question_toks[z]
            for z,tok_z in enumerate(db_str.split(" ")):
                token_list[start_idx + z]  = [SToken(text=tok_z),token_list[start_idx + z][1]]
                question_toks[start_idx + z]  = SToken(text=tok_z)
                sql['db_match'][start_idx + z] = [ [dbm[0],[len_start,len_end]] for dbm in sql['db_match'][start_idx + z]]
        elif db_str != " ".join([token_list[sii][0].text for sii in range(start_idx,start_idx+db_str.count(" ")+1)]):
            for sii, db_tok in zip(range(start_idx,start_idx+db_str.count(" ")+1), db_str.split(" ")):
                token_list[sii]  = [SToken(text=db_tok),token_list[start_idx][1]]
                question_toks[sii]  = SToken(text=db_tok)

    question_toks = [tok[0] for tok in token_list]
    all_db_str = []
    db_ = DBEngine.new_db(schema) 
    offset = 0

    for start_idx,tok in enumerate(question_toks):
        start_idx += offset
        if len(tok.text) <= 2 and not tok.text.isupper():
            continue

        if start_idx == 0 or question_toks[start_idx-1].text in ["?","."]:
            continue

        if (not tok.text.islower() or get_punctuation_word(question_toks,start_idx)) and not sql['db_match'][start_idx] and not str_is_num(tok.text) and not col_match[start_idx] and not sql['table_match'][start_idx] and len(tok.text)>3:
            if re.fullmatch(r".*(start|end)(.){0,5}with.*", sql["question"]):
                continue
            elif re.fullmatch(r"(.*?(contain|includ|with\s|without\s|have\s|has\s|having\s|had\s|like|\sas\s)|as\s).*(substring|string|char\s|chars\s|character\s|letter|word|phrase|prefix|suffix).*", sql["question"]) or re.fullmatch(r".*(contain|includ|like)(.){0,4}(').*?(').*", sql["question"]) or re.fullmatch(r".*(hav|includ|has|with)(.){0,4}((an|a|the)\s){0,1}(').*?(')\sin(\s|\?|\.).*", sql["question"]):
                continue
            elif re.fullmatch(r".*(with|contain|contains|containing|contained|include|including|includes|included|includ|includs|have|had|has|having|like|likes)\s([A-Z]){1,10}(\s|\.|\?){0,1}.*", sql["question"]):
                if not re.fullmatch(r".*(with|contain|contains|containing|contained|include|including|includes|included|includ|includs|have|had|has|having|like|likes)\s([A-Z]){1,10}([a-z]){0,10}\s([A-Z]){1,10}([a-z]){0,10}\s(\.|\?){0,1}.*", sql["question"]):
                    continue

            all_str = db_.get_all_db_string()

            words = [(tok.text,len(tok.text),tok.text[0].lower())]
            if start_idx + 1  < len(question_toks):
                words.append((tok.text + " " + question_toks[start_idx + 1].text,len(tok.text + " " + question_toks[start_idx + 1].text),tok.text[0].lower()))
            if start_idx + 2  < len(question_toks):
                words.append( (tok.text + " " + question_toks[start_idx + 1].text+ " " +question_toks[start_idx + 2].text, len(tok.text + " " + question_toks[start_idx + 1].text+ " " +question_toks[start_idx + 2].text),tok.text[0].lower()) )
            if start_idx > 0 and sql['db_match'][start_idx-1]:
                words.append( (question_toks[start_idx - 1].lower_ + " " + tok.lower_, len(question_toks[start_idx - 1].text + " " + tok.text),question_toks[start_idx - 1].text[0].lower()) )

            bk_bool = False

            for ass in all_str:
                for col_str in ass:
                    if not col_str[0] or type(col_str[0]) != str:
                        continue
                    word_num = col_str[0].count(" ")
                    if start_idx + word_num >= len(question_toks) or word_num > 2:
                        continue
                    tok_l = len(col_str[0])
                    if col_str[0][0].lower() == words[word_num][2] and (tok_l == words[word_num][1] or tok_l+1 == words[word_num][1] or tok_l-1 == words[word_num][1]):
                        sw = switch_tok_match(words[word_num][0],col_str[0])
                        edit_dis = editdistance.eval(words[word_num][0],col_str[0])
                        if sw or edit_dis <= 1 or (tok_l>=9 and edit_dis <= 2):
                            if edit_dis == 2:
                                print(words[word_num][0])
                                print(col_str[0])
                            for k in range(word_num+1):
                                question_toks[start_idx+k] = SToken(text=col_str[0].split(" ")[k])
                                token_list[start_idx+k] = [SToken(text=col_str[0].split(" ")[k]),token_list[start_idx][1]]
                            res = datebase_match_tables(schema,question_toks[start_idx],start_idx,question_toks,[-1],True)
                            if res:
                                res = [ress[0] for ress in res]
                                max_res = -1
                                for ri in res:
                                    if max_res < max(ri[1]):
                                        max_res = max(ri[1])
                                for ri in range(len(res)-1,-1,-1):
                                    if max_res != max(res[ri][1]):
                                        del res[ri]
                                for z in range(res[0][1][0],res[0][1][1]+1):
                                    sql["db_match"][z] = res
                            bk_bool = True
                            break
                    elif not bk_bool and word_num == 1 and len(words) == 3:
                        word_num += 1
                        if col_str[0].lower() == words[word_num][0]:
                            res = datebase_match_tables(schema,question_toks[start_idx-1],start_idx-1,question_toks,[-1],True)
                            if res:
                                res = [ress[0] for ress in res]
                                max_res = -1
                                for ri in res:
                                    if max_res < max(ri[1]):
                                        max_res = max(ri[1])
                                for ri in range(len(res)-1,-1,-1):
                                    if max_res != max(res[ri][1]):
                                        del res[ri]
                                for z in range(res[0][1][0],res[0][1][1]+1):
                                    sql["db_match"][z] = res
                            bk_bool = True
                            break
                    if bk_bool:
                        break

        if not sql['db_match'][start_idx] and (tok.lower_ in COUNTRYS_DICT.keys() or ((start_idx + 1) < len(question_toks) and tok.lower_ + " " + question_toks[start_idx+1].lower_ in COUNTRYS_DICT.keys())) :
            table_idxs = get_all_table(sql["table_match"],col_match,sql["db_match"],schema)
            if datebase_match_tables(schema,question_toks[start_idx],start_idx,question_toks,table_idxs,True):
                continue
            match = datebase_match_tables(schema,question_toks[start_idx-1],start_idx-1,question_toks,table_idxs,True)
            if start_idx > 1 and match and match[0][0][1][1] >= start_idx:
                continue
            match = datebase_match_tables(schema,question_toks[start_idx-2],start_idx-2,question_toks,table_idxs,True)
            if start_idx > 2 and match and match[0][0][1][1] >= start_idx:
                continue
            if tok.lower_ in COUNTRYS_DICT.keys():
                key = tok.lower_
            else:
                key = tok.lower_ + " " + question_toks[start_idx+1].lower_

            res = None
            for c in COUNTRYS_DICT[key]:
                res = datebase_match_tables(schema,SToken(text=c),0,[SToken(text=c)],table_idxs,True)
                if res:
                    k_space = key.count(" ")
                    c_space = c.count(" ")
                    if k_space < c_space:
                        for z in range(c_space-k_space):
                            token_list.insert(start_idx+1,[])
                            sql['db_match'].insert(start_idx+1,[])
                            sql['question_tag'].insert(start_idx+1,sql['question_tag'][start_idx])
                            sql['question_entt'].insert(start_idx+1,sql['question_entt'][start_idx])
                            sql['table_match'].insert(start_idx+1,[])
                            sql['question_dep']["data"].insert(start_idx+1,[])
                            col_match.insert(start_idx+1,[])
                            question_toks.insert(start_idx+1,[])
                    elif k_space > c_space:
                        for z in range(k_space-c_space):
                            del token_list[start_idx+1]
                            del sql['db_match'][start_idx+1]
                            del sql['question_tag'][start_idx+1]
                            del sql['question_entt'][start_idx+1]
                            del sql['table_match'][start_idx+1]
                            del sql['question_dep']["data"][start_idx+1]
                            del col_match[start_idx+1]
                            del question_toks[start_idx+1]

                    res = [resss  for ress in res for resss in ress]
                    for r_i in range(len(res)):
                        res[r_i][1][1] = c.count(" ")

                    for z in range(c_space+1):
                        tmp = c.split(" ")[z]
                        token_list[start_idx+z] = [SToken(text=tmp[0].upper() + tmp[1:]),token_list[start_idx][1]]
                        question_toks[start_idx+z] = SToken(text=tmp[0].upper() + tmp[1:])
                        sql["db_match"][start_idx+z] = res
                    break

        if tok.text in ["male","female","females","males"] and not sql['db_match'][start_idx] and not col_match[start_idx] and not sql['table_match'][start_idx]:
            add_db = True
            for _i,qtok in enumerate(question_toks):
                if qtok.lower_ == "f" and sql['db_match'][_i] and tok.text in ["female","females"]:
                    add_db = False
                elif qtok.lower_ == "m" and sql['db_match'][_i] and tok.text in ["male","males"]:
                    add_db = False
            if add_db:
                res = []
                for i,col in enumerate(schema.column_tokens_lemma_str):
                    if (schema.column_types[i] == "boolean" or col in ["sex | gender","gender | sex"]) and ("sex" in col or "male" in col or "female" in col):
                        res.append([i,[start_idx,start_idx]])
                sql['db_match'][start_idx] = res

        elif start_idx > 0 and lstem.stem(tok.text) in ["jan","feb","mar","apr", "may","jun","jul","aug", "sep","oct","nov","dec"] and not sql['db_match'][start_idx] and (question_toks[start_idx+1].text.isdigit() or question_toks[start_idx+1].text[:-2].isdigit() or question_toks[start_idx-1].text.isdigit() or question_toks[start_idx-1].text[:-2].isdigit()) and start_idx + 3 < len(question_toks) and not sql['db_match'][start_idx]:
            date = 0
            year_offset = 2
            start_dt = start_idx-1
            gold_dt = ""
            if question_toks[start_idx-1].text.isdigit():
                date = question_toks[start_idx-1].text
            elif  question_toks[start_idx-1].text[:-2].isdigit():
                date = question_toks[start_idx-1].text[:-2]
            if not date:
                year_offset = 3
                start_dt = start_idx
                if question_toks[start_idx+1].text.isdigit():
                    date = question_toks[start_idx+1].text
                elif  question_toks[start_idx+1].text[:-2].isdigit():
                    date = question_toks[start_idx+1].text[:-2]
            if question_toks[start_idx + year_offset].text.isdigit():
                year = question_toks[start_idx + year_offset].text
                end_dt = start_idx + year_offset
            elif question_toks[start_idx + year_offset].text == "," and start_idx + year_offset + 1 < len(question_toks) and question_toks[start_idx + year_offset + 1].text.isdigit():
                year = question_toks[start_idx + year_offset + 1].text
                end_dt = start_idx + year_offset + 1
            else:
                year = 0
            if year and date:
                month = str(["jan","feb","mar","apr", "may","jun","jul","aug", "sep","oct","nov","dec"].index(lstem.stem(tok.text)) + 1)
                datetime_set = []
                datetime = year + "-"
                datetime_set.append(datetime+month)
                if len(month) == 1:
                    datetime_set.append(datetime+"0"+month)
                gold_dt = datetime_set[-1]
                for j,d in enumerate(datetime_set):
                    datetime_set[j] = d + "-" + date
                if len(date) == 1:
                    gold_dt = gold_dt + "-0" + date
                    for d in copy.deepcopy(datetime_set):
                        datetime_set.append(d[:-1]+"0"+d[-1:])
                else:
                    gold_dt = gold_dt + "-" + date
                for ds in copy.deepcopy(datetime_set):
                    datetime_set.append(ds.replace('-','/'))
                    ymd = ds.split('-')
                    datetime_set.append(ymd[1]+"/"+ymd[2]+"/"+ymd[0])
                table_idxs = get_all_table(sql["table_match"],col_match,sql["db_match"],schema)
                res = None
                for c in datetime_set:
                    res = datebase_match_tables(schema,SToken(text=c),0,[SToken(text=c)],table_idxs,True)
                    if res:
                        break
                for z in range(end_dt,start_dt,-1):
                    del token_list[z]
                    del sql['db_match'][z]
                    del sql['question_tag'][z]
                    del sql['question_entt'][z]
                    del sql['table_match'][z]
                    del sql['question_dep']["data"][z]
                    del col_match[z]
                    del question_toks[z]
                sql['question_entt'][start_dt] = "DATE"
                question_toks[start_dt] = SToken(text=c)
                if res:
                    question_toks[start_dt] = SToken(text=c)
                    token_list[start_dt] = [SToken(text=c),token_list[start_dt][1]]
                    sql['db_match'][start_dt] = [resss  for ress in res for resss in ress]
                else:
                    question_toks[start_dt] = SToken(text=gold_dt)
                    token_list[start_dt] = [SToken(text=gold_dt),token_list[start_dt][1]]
        elif start_idx > 0 and not sql['db_match'][start_idx-1] and sql['db_match'][start_idx] and sql['db_match'][start_idx][0][1][1] > sql['db_match'][start_idx][0][1][0]:
            correct_multiple_tokens(schema,sql,start_idx,question_toks,token_list)
        elif start_idx > 0 and sql['db_match'][start_idx-1] != sql['db_match'][start_idx] and sql['db_match'][start_idx] and sql['db_match'][start_idx][0][1][1] == sql['db_match'][start_idx][0][1][0]:
            db_str = get_database_string(schema, sql['db_match'][start_idx][0][0], question_toks[start_idx:start_idx + sql['db_match'][start_idx][0][1][1] - sql['db_match'][start_idx][0][1][0] + 1])
            if not db_str and token_list[start_idx][0].text in {"male":"m","female":"f","females":"f","males":"m"}:
                db_str = get_database_string(schema, sql['db_match'][start_idx][0][0], [SToken(text={"male":"m","female":"f","females":"f","males":"m"}[token_list[start_idx][0].text])])
            assert db_str
            if db_str != token_list[start_idx][0].text:
                token_list[start_idx]  = [SToken(text=db_str),token_list[start_idx][1]]
                question_toks[start_idx]  = SToken(text=db_str)
        elif start_idx > 0 and start_idx + 2 < len(question_toks) and not sql['db_match'][start_idx] and token_list[start_idx][0].text.isalpha() and (not token_list[start_idx][0].text.islower() or (token_list[start_idx-1][0].text == "'" and get_punctuation_word(question_toks,start_idx,True))):
            for or_i,tok in enumerate(sql['or_question']):
                if tok.text == token_list[start_idx][0].text and or_i >= start_idx:
                    res = datebase_match_tables(schema,tok,or_i,sql['or_question'],[-1],True)
                    if res:
                        add_word_offset = 0
                        for res_idx in range(res[0][0][1][0],res[0][0][1][1]+1):
                            if sql['or_question'][res_idx].text != token_list[start_idx+add_word_offset][0].text:
                                token_list.insert(start_idx+add_word_offset,[SToken(text=sql['or_question'][res_idx].text),token_list[start_idx+add_word_offset][1]])
                                sql['db_match'].insert(start_idx+add_word_offset,[resss  for ress in res for resss in ress])
                                sql['question_tag'].insert(start_idx+add_word_offset,sql['question_tag'][start_idx+add_word_offset])
                                sql['question_entt'].insert(start_idx+add_word_offset,sql['question_entt'][start_idx+add_word_offset])
                                sql['table_match'].insert(start_idx+add_word_offset,[])
                                sql['question_dep']["data"].insert(start_idx+add_word_offset,[])
                                col_match.insert(start_idx+add_word_offset,[])
                                question_toks.insert(start_idx+add_word_offset,SToken(text=sql['or_question'][res_idx].text))
                            sql['db_match'][start_idx+add_word_offset] = [resss  for ress in res for resss in ress]
                            add_word_offset += 1
                        correct_multiple_tokens(schema,sql,start_idx,question_toks,token_list)
                    break
    return token_list,sql,col_match


        

def db_continue_correction(token_list, sql, schema):
    def get_db_col(db_m):
        cols = []
        for dbm in db_m:
            cols.append(dbm[0])
        return set(cols)
    
    def get_col_tables(cols):
        tbs = []
        for c in cols:
            tbs.append(schema.column_names_original[c][0])
        return set(tbs)

    for (start_idx,tok),db_m in zip(enumerate(token_list),sql["db_match"]):
        offset = 0
        if start_idx > 1 and sql["db_match"][start_idx] and sql["db_match"][start_idx-1] and tok[1] == token_list[start_idx-1][1] and get_db_col(sql["db_match"][start_idx]) != get_db_col(sql["db_match"][start_idx-1]) and tok[0].text.islower() == token_list[start_idx-1][0].text.islower():
            offset = 1
        elif start_idx > 2 and sql["db_match"][start_idx] and sql["db_match"][start_idx-2] and not sql["db_match"][start_idx-1] and token_list[start_idx-1][0].text in ["and","or"] and tok[1] == token_list[start_idx-2][1] and get_db_col(sql["db_match"][start_idx]) != get_db_col(sql["db_match"][start_idx-2]) and tok[0].text.islower() == token_list[start_idx-1][0].text.islower():
            offset = 2
        if offset:
            offset += sql["db_match"][start_idx-offset][0][1][1] - sql["db_match"][start_idx-offset][0][1][0]
            tbs1 = get_col_tables(get_db_col(sql["db_match"][start_idx]))
            tbs2 = get_col_tables(get_db_col(sql["db_match"][start_idx-offset]))
            if tbs1 != tbs2:
                tbs = tbs1.union(tbs2)
                question_toks = [tok[0] for tok in token_list]
                res1 = datebase_match_tables(schema,question_toks[start_idx],start_idx,question_toks,tbs,True)
                res2 = datebase_match_tables(schema,question_toks[start_idx-offset],start_idx-offset,question_toks,tbs,True)
                res1 = [r for rs in res1 for r in rs]
                res2 = [r for rs in res2 for r in rs]
                res1_col = [rs[0] for rs in res1 ]
                res2_col = [rs[0] for rs in res2 ]
                res1_tb = get_col_tables(res1_col)
                res2_tb = get_col_tables(res2_col)
                update_db_m = False
                for tb in res1_tb:
                    if tb in res2_tb:
                        update_db_m = True
                if update_db_m:
                    for i in range(res1[0][1][1]-res1[0][1][0]+1):
                        sql["db_match"][start_idx+i]=[]
                        for rs in res1:
                            if schema.column_names_original[rs[0]][0] in res1_tb and schema.column_names_original[rs[0]][0] in res2_tb:
                                sql["db_match"][start_idx+i].append(rs)
                    for i in range(res2[0][1][1]-res2[0][1][0]+1):
                        sql["db_match"][start_idx-offset+i]=[]
                        for rs in res2:
                            if schema.column_names_original[rs[0]][0] in res1_tb and schema.column_names_original[rs[0]][0] in res2_tb:
                                sql["db_match"][start_idx-offset+i].append(rs)
    return sql

def others_analyze(sentences, table_matchs, list_idxs, schema, sub_q, q_sql, select_type=False, in_db_match=None, final_select=False, no_pattern=False):
    pattern_type = []
    others = []
    pattern_unknow = []
    others_token_pattern = []
    for sentence, table_match, list_idx in zip(sentences,table_matchs,list_idxs):
        tables = look_for_table_idx(sub_q, list_idx, 1, schema)
        ts = TokenString(None,sub_q.question_tokens[list_idx])
        pattern_unknow.append([pattern_recomand(ts,table_match,sub_q.col_match[list_idx],sub_q.sequence_entt[list_idx],sub_q.db_match[list_idx],schema,tables)[0:2],None])
        others_token_pattern.append(pattern_unknow[-1][0][1])
        sub_q.pattern_tok[list_idx] = pattern_unknow[-1][0][1]
        if len(others_token_pattern[-1]) != len(table_match):
            print("Pattern Error!!!!!!!!!!!!!!!!!")
    return others,pattern_type,others_token_pattern


def is_none_type(pattern_toks,words):
    for pt,word in zip(pattern_toks,words):
        if pt in {"AGG","BCOL", "NUM","YEAR","DATE","DB","PDB","SDB","UDB","NOT","GRSM","SGRSM","SM_GRSM","SM_SGRSM","GR_GRSM","GR_SGRSM","GR_SJJS","SM_SJJS","SJJS","SM_JJS","GR_JJS","JJS","order","sort"}:
            if pt == "AGG" and word.lemma_ in ["average","number","total","mean","avg"]:
                continue
            return False
    return True

def combine_none_subquestion(token_list,sql,schema,sentence_num,col_match):
    sq,qsql = generate_sq_qsql(token_list,sql,schema,sentence_num)
    select_list,select_table,select_list_idx,select_db_idx = sq.sentence_combine(type_offset=1)
    others_list,others_table,others_list_idx,others_db_idx = sq.sentence_combine(combine_type=1,type_offset=1)
    for (i,cm),oi in zip(enumerate(sq.col_match),sq.original_idx):
        sq.col_match[i] = [col_match[idx] for idx in oi]
    others,_,select_token_pattern = others_analyze(select_list,select_table, select_list_idx, schema,sq,qsql,True,select_db_idx,no_pattern=True)
    others,pattern_type,others_token_pattern = others_analyze(others_list,others_table, others_list_idx, schema,sq,qsql,in_db_match=others_db_idx)
    
    for (j,q_type),patterns in zip(enumerate(sq.sub_sequence_type),sq.pattern_tok):
        if j > 0 and len(sq.question_tokens[j]) == 1:
            sq.sub_sequence_type[j] = sq.sub_sequence_type[j-1]
            for idx in sq.original_idx[j]:
                token_list[idx][1] = sq.sub_sequence_type[j]
        elif q_type > 0 and sq.question_tokens[j][-1].text not in [",","?","."] and j + 1 < len(sq.sub_sequence_type) and is_none_type(patterns,sq.question_tokens[j]):
            if (sq.pattern_tok[j][0] == "WP" or sq.question_tokens[j][0].lemma_ in ["can","be","have","and","or",",","but"] or (j>1 and sq.question_tokens[j-1][-1].lemma_ in ["and","or",","]) or sq.question_tokens[j][0].tag_.startswith("V")) \
                and  (sq.pattern_tok[j+1][0] == "IN" or sq.question_tokens[j+1][0].lower_ == "in" or len(sq.pattern_tok[j])<=3 or (j+2 == len(sq.pattern_tok) and is_none_type(sq.pattern_tok[j+1],sq.question_tokens[j+1]))) \
                and sq.question_tokens[j][0].lower_ not in ["for", "along","together"] \
                and sq.question_tokens[j+1][0].lower_ not in ["but","for","along","together",","] and (sq.question_tokens[j][-1].lower_ not in ["but","and",",","."] or (sq.question_tokens[j][-1].lower_ == "and" and len(sq.question_tokens[j])<=3 and (j<=0 or sq.sub_sequence_type[j-1] !=sq.sub_sequence_type[j+1])) ) \
                and (sq.sub_sequence_type[j+1] >= 0 or len(sq.question_tokens[j])<=3) and not (len(sq.question_tokens[j])<=4 and "each" in patterns):
                sq.sub_sequence_type[j] = sq.sub_sequence_type[j+1]
                for idx in sq.original_idx[j]:
                    token_list[idx][1] = sq.sub_sequence_type[j]
                continue
            elif "DB" in sq.pattern_tok[j+1] or "UDB" in sq.pattern_tok[j+1] or "PDB" in sq.pattern_tok[j+1]:
                match = False
                for dbms in sq.db_match[j+1]:
                    if dbms:
                        for dbm in dbms:
                            col = dbm[0]
                            for cols in sq.col_match[j]:
                                if cols and col in cols[0]:
                                    match = True
                                    break
                if not match and ("UDB" in sq.pattern_tok[j+1] or "PDB" in sq.pattern_tok[j+1]):
                    if sq.col_match[j][-1]:
                        match = True
                if match:
                    sq.sub_sequence_type[j] = sq.sub_sequence_type[j+1]
                    for idx in sq.original_idx[j]:
                        token_list[idx][1] = sq.sub_sequence_type[j]
                    continue
            elif "DATE" in sq.pattern_tok[j+1] or "YEAR" in sq.pattern_tok[j+1] or "NUM" in sq.pattern_tok[j+1]:
                match = False
                if "DATE" in sq.pattern_tok[j+1] or "YEAR" in sq.pattern_tok[j+1]:
                    for cols in sq.col_match[j]:
                        if cols:
                            for col in cols[0]:
                                if schema.column_types[col] in ["time","year"]:
                                    match = True
                                    break
                elif sq.col_match[j][-1]:
                    for col in sq.col_match[j][-1][0]:
                        if schema.column_types[col] in ["number"]:
                            match = True
                if match:
                    sq.sub_sequence_type[j] = sq.sub_sequence_type[j+1]
                    for idx in sq.original_idx[j]:
                        token_list[idx][1] = sq.sub_sequence_type[j]
                    continue
            if j > 0 and sq.table_match[j-1].count([]) == len(sq.table_match[j-1]) and sq.table_match[j].count([]) != len(sq.table_match[j]):
                match = True if sq.col_match[j-1].count([]) == len(sq.col_match[j-1]) else False
                for ts in sq.table_match[j]:
                    if ts:
                        for cols in sq.col_match[j-1]:
                            if cols:
                                for col in cols[0]:
                                    if schema.column_tokens_table_idx[col] in ts:
                                        match = True
                                        break
                if match:
                    sq.sub_sequence_type[j] = sq.sub_sequence_type[j-1]
                    for idx in sq.original_idx[j]:
                        token_list[idx][1] = sq.sub_sequence_type[j]
        elif q_type > 0 and j > 0 and "order" in patterns and sq.question_tokens[j][0].tag_ == "IN" and ("sort" in sq.pattern_tok[j-1] or "order" in sq.pattern_tok[j-1]) and (sq.pattern_tok[j-1][0] not in ["order","sort"] or j > 1):
            sq.sub_sequence_type[j] = sq.sub_sequence_type[j-1]
            for idx in sq.original_idx[j]:
                token_list[idx][1] = sq.sub_sequence_type[j]
    return token_list
