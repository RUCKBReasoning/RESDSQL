import pickle,json
import spacy 
import editdistance
import copy
import argparse
from spacy.symbols import ORTH, LEMMA

from natsql2sql.preprocess.question_repair import question_repair
from natsql2sql.preprocess.sentence_analyse import db_correction,db_continue_correction,sentence_cut,reshap_token,sentence_dump,merge_noun_chunks,re_analyse_sentence,correct_121_pattern,correct_special_pattern,pattern_sentence_analyse,final_correct_special_pattern,easy_cut,two_setence_analyse,select_split,reset_uncontinue_type,pattern_generate_col,anaylse_punctuate,special_word_modify,combine_none_subquestion
from natsql2sql.preprocess.TokenString import get_spacy_tokenizer,TokenString,SToken
from natsql2sql.preprocess.table_match import return_table_name
from natsql2sql.preprocess.Schema_Token import Schema_Token
from natsql2sql.preprocess.sq import SubQuestion,QuestionSQL
from natsql2sql.preprocess.utils import construct_select_data
from natsql2sql.preprocess.db_match import datebase_match_tables,get_database_col
from natsql2sql.preprocess.match import COUNTRYS_DICT


def special_replace(str_):
    str_ = str_.replace("（","(")
    str_ = str_.replace("）",")")
    str_ = str_.replace("”","\"")
    str_ = str_.replace("“","\"")
    str_ = str_.replace("“","\"")
    str_ = str_.replace("`","'")
    str_ = str_.replace(" '' "," ' ")
    str_ = str_.replace(" ' ' "," ' ")
    
    if str_[:7].lower() == "please ":
        str_ = str_[7:]
    str_ = str_.replace("Please ","")
    str_ = str_.replace("Sort the list of ","Sort ")
    str_ = str_.replace("Sort the list ","Sort ")
    if str_[:5].lower() == "list ":
        str_ = "Show " + str_[5:]
    str_ = str_.replace("List ","Show ")
    if str_[1:9].lower() == "n which ":
        str_ = "Which " + str_[9:]

    str_ = str_.replace("In which ","Which ")
    str_ = str_.replace(" , in which "," , which ")
    str_ = str_.replace(" and in which "," and what ")

    str_ = str_.replace("At which ","Which ")
    str_ = str_.replace(" , at which "," , which ")
    str_ = str_.replace(" and at which "," and what ")

    str_ = str_.replace("On which ","Which ")
    str_ = str_.replace(" , on which "," , Which ")
    str_ = str_.replace(" and on which "," and what ")

    str_ = str_.replace("Which of the ","Which ")
    str_ = str_.replace("Which of ","Which ")
    str_ = str_.replace("On what ","What ")
    str_ = str_.replace("In what ","What ")
    str_ = str_.replace(" in what "," what ")
    str_ = str_.replace("From which ","Which ")


    if str_[:9].lower() == "which of ":
        str_ = "Which " + str_[9:]

    str_ = str_.replace("How much does ","What is ")
    str_ = str_.replace("How much do ","What is ")
    str_ = str_.replace("How much did ","What was ")
    str_ = str_.replace(" how much does "," what is ")
    str_ = str_.replace(" how much do "," what is ")
    str_ = str_.replace(" how much did "," what was ")
    str_ = str_.replace("How much is ","What is the price of ")
    str_ = str_.replace("How much was ","What was the price of ")
    str_ = str_.replace(" how much is "," what is the price of ")
    str_ = str_.replace(" how much was "," what was the price of ")
    str_ = str_.replace("How much ","What is ")
    str_ = str_.replace(" how much "," what is ")


    str_ = str_.replace(" numbers of "," number of ")
    str_ = str_.replace(" count of "," number of ")
    str_ = str_.replace(" the count "," the number ")
    str_ = str_.replace(" total number of "," number of ")
    str_ = str_.replace(" number of records of "," number of ")


    str_ = str_.replace(" all of "," ")
    str_ = str_.replace(" as well as "," and ")
    str_ = str_.replace(" along with "," and ")
    str_ = str_.replace(" in total "," ")


    str_ = str_.replace(" corresponding to "," for ")
    str_ = str_.replace(" correspond to "," for ")
    str_ = str_.replace(" corresponded to "," for ")
    str_ = str_.replace(" corresponding "," ")


    str_ = str_.replace(" for all the different "," for each ")
    str_ = str_.replace(" for different "," for each ")
    str_ = str_.replace(" for the different "," for each ")
    str_ = str_.replace(" of different "," of each ")
    str_ = str_.replace(" of the different "," of each ")
    str_ = str_.replace(" different and every "," each ")
    str_ = str_.replace(" does different "," for each ")
    str_ = str_.replace(" does the different "," for each ")
    str_ = str_.replace(" among different "," for each ")
    str_ = str_.replace(" among the different "," for each ")
    str_ = str_.replace(" in different "," for each ")
    str_ = str_.replace(" in the different "," for each ")
    str_ = str_.replace(" with different "," for each ")
    str_ = str_.replace(" with the different "," for each ")
    str_ = str_.replace(" by different "," by each ")
    str_ = str_.replace(" each different "," each ")


    str_ = str_.replace(" each and every "," each ")
    str_ = str_.replace(" does each "," for each ")
    str_ = str_.replace(" among each "," for each ")
    str_ = str_.replace(" in each "," for each ")
    str_ = str_.replace(" with each "," for each ")
    str_ = str_.replace(" each number of "," each ")
    str_ = str_.replace(" number of each "," number of ")
    str_ = str_.replace("Show each ","Show ")
    str_ = str_.replace("Find each ","Find ")
    str_ = str_.replace("Return each ","Return ")
    str_ = str_.replace("Count each ","Count ")
    str_ = str_.replace("Give each ","Give ")
    str_ = str_.replace("Compute each ","Compute ")
    str_ = str_.replace("Tell each ","Tell ")
    str_ = str_.replace(" has each "," for each ")
    str_ = str_.replace(" have each "," for each ")
    str_ = str_.replace(" for each ."," .")
    str_ = str_.replace(" for each ?"," ?")
    str_ = str_.replace(" for each have ."," .")
    str_ = str_.replace(" for each has ."," .")
    str_ = str_.replace(" for each had ."," .")
    str_ = str_.replace(" for each have ?"," ?")
    str_ = str_.replace(" for each has ?"," ?")
    str_ = str_.replace(" for each had ?"," ?")
    str_ = str_.replace("What are each ","What are ")
    str_ = str_.replace("What is each ","What is ")
    str_ = str_.replace("What was each ","What was ")
    str_ = str_.replace("What were each ","What were ")
    str_ = str_.replace(" what are each "," what are ")
    str_ = str_.replace(" what is each "," what is ")
    str_ = str_.replace(" what was each "," what was ")
    str_ = str_.replace(" what were each "," what were ")
    str_ = str_.replace("Report ","Show ")


    str_ = str_.replace(" from each "," for each ")

    ####################################################

    if str_.lower().startswith("how many "):
        str_ = str_.replace(" are listed ?"," are there ?")
        str_ = str_.replace(" is listed ?"," is there ?")
    str_ = str_.replace(" how many had "," how many ")
    str_ = str_.replace(" how many has "," how many ")
    str_ = str_.replace(" how many have "," how many ")
    str_ = str_.replace("How many had "," how many ")
    str_ = str_.replace("How many has "," how many ")
    str_ = str_.replace("How many have "," how many ")
    
    str_ = str_.replace(" that of any "," any ")
    str_ = str_.replace(" across all "," of ")
    str_ = str_.replace(" cross all "," of ")

    str_ = str_.replace(" ; "," . ")
    str_ = str_.replace(" . Order by "," , order by ")
    str_ = str_.replace(" ? Order by "," , order by ")
    str_ = str_.replace(" by order "," , order by ")

    str_ = str_.replace(" together with "," and ")
    str_ = str_.replace(" , sorted "," , which sorted ")
    str_ = str_.replace(" , and sort "," , which sorted ")
    str_ = str_.replace(" and sort "," , which sorted ")
    str_ = str_.replace("Sort each ","Sort ")
    str_ = str_.replace("Order each ","Order ")
    str_ = str_.replace("Order by each ","Order by ")
    str_ = str_.replace(" sort each "," sort ")
    str_ = str_.replace(" order each "," order ")
    str_ = str_.replace(" order by each "," order ")
    str_ = str_.replace(" sorted each "," sorted ")
    str_ = str_.replace(" ordered each "," ordered ")
    str_ = str_.replace(" ordered by each "," ordered ")
    str_ = str_.replace("Sort the each ","Sort ")
    str_ = str_.replace("Order the each ","Order ")
    str_ = str_.replace("Order by the each ","Order by ")
    str_ = str_.replace(" sort the each "," sort ")
    str_ = str_.replace(" order the each "," order ")
    str_ = str_.replace(" order by the each "," order ")
    str_ = str_.replace(" sorted the each "," sorted ")
    str_ = str_.replace(" ordered the each "," ordered ")
    str_ = str_.replace(" ordered by the each "," ordered ")


    str_ = str_.replace(" alphaetical "," alphabetical ")
    str_ = str_.replace(" as long as "," as ")
    str_ = str_.replace(" prior to "," before ")
    str_ = str_.replace(" in terms of "," of ")
    str_ = str_.replace(" listed by "," ordered by ")
    str_ = str_.replace(" listed descend"," ordered by descend")
    str_ = str_.replace(" listed ascend"," ordered by ascend")
    str_ = str_.replace(" listed alphabetic"," ordered by alphabetic")
    str_ = str_.replace(" listed in descend"," ordered by descend")
    str_ = str_.replace(" listed in ascend"," ordered by ascend")
    str_ = str_.replace(" listed in alphabetic"," ordered by alphabetic")
    str_ = str_.replace(" alphabetic "," alphabetical ")
    str_ = str_.replace(" alphabetically "," alphabetical ")
    str_ = str_.replace(" lexicographical "," alphabetical ")
    str_ = str_.replace(" lexicographically "," alphabetical ")
    str_ = str_.replace(" lexicographic "," alphabetical ")
    str_ = str_.replace(" decreasing "," descending ")
    str_ = str_.replace(" increasing "," ascending ")


    str_ = str_.replace("Show in alphabetical order ","In alphabetical order , show ")
    str_ = str_.replace("Show in decreasing order ","In decreasing order , show ")
    str_ = str_.replace("Show in ascending order ","In ascending order , show ")
    str_ = str_.replace("Show from which ","Show ")


    str_ = str_.replace("Return in alphabetical order ","In alphabetical order , show ")
    str_ = str_.replace("Return in decreasing order ","In decreasing order , show ")
    str_ = str_.replace("Return in ascending order ","In ascending order , show ")
    str_ = str_.replace("Return from which ","Show ")


    str_ = str_.replace(" belongs to "," belonged to ")
    str_ = str_.replace(" belong to "," belonged to ")
    str_ = str_.replace(" presently "," ")
    str_ = str_.replace(" years old "," age ")
    str_ = str_.replace(" year old "," age ")
    str_ = str_.replace(" age or old "," shinier age ")
    str_ = str_.replace(" age or older "," shinier age ")
    str_ = str_.replace(" age or young "," uglier age ")
    str_ = str_.replace(" age or younger "," uglier age ")
    str_ = str_.replace(" age age "," age ")    

    str_ = str_.replace(" first letter is "," start with ")   
    str_ = str_.replace(" most recent "," latest ")   
    str_ = str_.replace(" most recently "," latest ")   

    str_ = str_.replace(" family name "," last name ")   
    str_ = str_.replace(" family names "," last names ")   
    str_ = str_.replace(" number of unique "," number of different ")   
    str_ = str_.replace("ow many unique ","ow many different ")   

    str_ = str_.replace(" , for all "," of all ")
    str_ = str_.replace(" for all "," of all ")
    str_ = str_.replace(" of the "," of ")
    str_ = str_.replace(" of all the "," of all ")
    str_ = str_.replace(" of a "," of ")


    str_ = str_.replace("How old ","What age ")
    str_ = str_.replace(" how old is "," what age is ")
    str_ = str_.replace(" how old are "," what age are ")
    
    str_ = str_.replace(" and that of "," and ")
    str_ = str_.replace(" at8least8one "," at least one ")

    str_ = str_.replace(" than that of some at least one "," than some ")
    str_ = str_.replace(" than that of a "," than a ")
    str_ = str_.replace(" than that of some "," than some ")
    str_ = str_.replace(" than that of all "," than all ")
    str_ = str_.replace(" than that of any "," than any ")
    str_ = str_.replace(" than that of "," than ")
    str_ = str_.replace(" at least once "," at least one ")
    str_ = str_.replace(" at least 1 "," at least one ")
    str_ = str_.replace(" at least a "," at least one ")
    str_ = str_.replace(" at least an "," at least one ")
    str_ = str_.replace(" one or more "," at least one ")
    str_ = str_.replace(" some at least one "," at least one ")
    str_ = str_.replace(" than at least one "," than a ")
    str_ = str_.replace(" in at least one "," in a ")

    return str_


def dump_language_feature(sentences):
    sent_data = {}
    sent_data['root'] = []
    sent_data['data'] = []
    for sent in sentences.sents:
        sent_data = sentence_dump(sent,sent_data)
    return sent_data


def question_repair_select_part(sql,token_list,schema):
    def select_word_repair(sentences, table_matchs, list_idxs, schema:Schema_Token, sub_q, q_sql):
        replace_old = []
        replace_new = []
        select,where,group,order = (None,None,None,None)
        sentence_num = len(sentences)
        re_value = []
        key_word = []
        for sentence,table_match,list_idx in zip(sentences, table_matchs, list_idxs):
            availble_sent,availble_table,availble_idx,original_sent,original_table,original_idx = construct_select_data(sentence, table_match, sub_q.original_idx[list_idx],schema)
            for s,t,i in zip(availble_sent,availble_table,availble_idx):
                if len(s) > 1 and len(s[-1]) == 1 and not t[-1][0]:# and not datebase_match_tables(schema,q_sql.question_tokens[i[-1][0]],i[-1][0],q_sql.question_tokens,[i for i in range(len(schema.table_tokens))],return_all_match = True):
                    for st in schema.table_tokens_text_str:
                        for tw in st.split(" "):
                            if s[-1][0] not in replace_old and editdistance.eval(s[-1][0],tw) < 2:
                                if not datebase_match_tables(schema,q_sql.question_tokens[i[-1][0]],i[-1][0],q_sql.question_tokens,[i for i in range(len(schema.table_tokens))],return_all_match = True): # db match in here can be run fast.
                                    print(s[-1][0]+"--->"+tw)
                                    replace_old.append(s[-1][0])
                                    replace_new.append(tw)
                                    break
        return replace_old,replace_new
        
    tokens = [i[0] for i in token_list]
    sq = SubQuestion(copy.deepcopy(sql["question"]),copy.deepcopy(sql["question_type"]),copy.deepcopy(sql["table_match"]),copy.deepcopy(sql["question_tag"]),copy.deepcopy(sql["question_dep"]),copy.deepcopy(sql["question_entt"]),dict())
    select_list,select_table,select_list_idx,_ = sq.sentence_combine()
    others_list,others_table,others_list_idx,_ = sq.sentence_combine(combine_type=1)
    qsql = QuestionSQL(sq,None,tokens)
    sq.tokenize(qsql)
    replace_old,replace_new = select_word_repair(select_list,select_table, select_list_idx, schema ,sq,qsql)
    return replace_old,replace_new



def final_cut(token_list,db_match,table_match,col_match,question,keep_original_question):
    sum_ = sum([tl[1] for tl in token_list])
    and_should_cut = False
    if sum_ == 0 and " and " in question:
        for tl,db,i in zip(reversed(token_list),reversed(db_match),range(len(db_match)-1,-1,-1)):
            if tl[0].text == "than":
                and_should_cut = True
            if tl[0].text == "and":
                if and_should_cut:
                    pass
                elif i+2 < len(db_match) and ((db_match[i+1] and not token_list[i+1][0].text.islower()) or (db_match[i+2] and not token_list[i+2][0].text.islower())) and i > 2 and ((db_match[i-1] and not token_list[i-1][0].text.islower()) or (db_match[i-2] and not token_list[i-2][0].text.islower()) or (db_match[i-3] and not token_list[i-3][0].text.islower())):
                    pass
                elif i+3 < len(db_match) and (db_match[i+3] and not token_list[i+3][0].text.islower()) and i > 2 and ((db_match[i-1] and not token_list[i-1][0].text.islower()) or (db_match[i-2] and not token_list[i-2][0].text.islower()) or (db_match[i-3] and not token_list[i-3][0].text.islower())):
                    pass
                else:
                    continue
                see_between = False
                for k in range(i-1,-1,-1):
                    if token_list[k][0].text == "and":
                        break
                    if token_list[k][0].text == "between":
                        see_between = True
                        break
                if not see_between:
                    for j in range(i,len(db_match)):
                        token_list[j][1] = 1
            if tl[0].text in ["with","for"]:
                if and_should_cut:
                    for j in range(i,len(db_match)):
                        token_list[j][1] = 1
                    and_should_cut = False
    if not keep_original_question:
        for tl,db,i in zip(reversed(token_list),reversed(db_match),range(len(db_match)-1,-1,-1)):
            if tl[0].text == "and": # prevent and cut the col to two separate column
                if i+2 < len(db_match) and i and col_match[i-1] and col_match[i+1] and col_match[i-1][0] == col_match[i+1][0] and len(col_match[i-1][0])==1:
                    token_list[i][0] = SToken(text="'s")
                elif i+2 < len(db_match) and i and table_match[i-1] and table_match[i+1] and table_match[i-1][0] == table_match[i+1][0] and len(table_match[i-1][0])==1:
                    token_list[i][0] = SToken(text="'s")
    if len(token_list) == len(db_match):
        for (i,tl),dbm in zip(enumerate(token_list),db_match):
            if dbm and i > 0 and token_list[i-1][1] != tl[1] and db_match[i-1] == dbm:
                start_type = tl[1]
                for j in range(i,len(token_list)):
                    if token_list[j][1] != start_type:
                        break
                    token_list[j][1] = token_list[i-1][1]
    return token_list


def generate_full_db_match(token_list,col_match,sql,schema):
    question_toks = [tok[0] for tok in token_list]
    re_list = []
    for start_idx,tok in enumerate(token_list):
        if (start_idx == 0 or (start_idx > 0 and sql['db_match'][start_idx-1] != sql['db_match'][start_idx])) and sql['db_match'][start_idx]:
            db_str = get_database_col(schema, sql['db_match'][start_idx], question_toks[start_idx:start_idx + sql['db_match'][start_idx][0][1][1] - sql['db_match'][start_idx][0][1][0] + 1])
            re_list.append(db_str)
        else:
            if sql['db_match'][start_idx]: 
                assert re_list[-1]
                re_list.append(re_list[-1])
            else:
                re_list.append([])
    return re_list


def preprocess_sql(sql_path,table_path,other_words=[],all_word=[], keep_original_question=True, use_pattern_generate_col=False):
    """
    1. repair question
    2. calc table match
    3. sentence cut
    """
    sqls = json.load(open(sql_path, 'r'))
    table_json = json.load(open(table_path, 'r'))
    table_dict = {}
    full_word = {}
    all_schema = {}
    for table in table_json:
        table_dict[table['db_id']] = table
        word_in_table = []
        for w in table['column_names']:
            word_in_table.extend(w[1].split(" "))
        for w in table['table_names']:
            word_in_table.extend(w.split(" "))
        word_in_table.extend(other_words)
        word_in_table = list(set(word_in_table))
        top_w_in_t = [w[:-1] if len(w) >=3 else w for w in word_in_table ]
        end_w_in_t = [w[1:] if len(w) >=3 else w for w in word_in_table ]
        full_word[table['db_id']] = [word_in_table,top_w_in_t,end_w_in_t]

    _tokenizer = get_spacy_tokenizer()
    nlp = _tokenizer.spacy
    
    table_dict = {}
    for table in table_json:
        table_dict[table["db_id"]] = table
        if not ONE_ID:
            all_schema[table["db_id"]] = Schema_Token(_tokenizer,None,table,None)

    fast_db_match_json = None

    if not keep_original_question:
        for i,sql in enumerate(sqls):
            if ONE_ID and i!=ONE_ID:
                continue
            if ONE_ID:
                all_schema[sql["db_id"]] = Schema_Token(_tokenizer,None,table_dict[sql["db_id"]],None)
            sql['or_question'] = [tok for tok in nlp(sql['question'])]
            qq = question_repair(nlp,sql['question'],all_word,full_word[sql['db_id']][0],full_word[sql['db_id']][1],full_word[sql['db_id']][2])
            sql['question'] =  " ".join([tok.text for tok in qq])
            sql['question'] = sql['question'].replace("'s '","' s '")
            sql['question'] = special_replace(sql['question'])
            sql['question'] = special_word_modify(sql['question'],all_schema[sql["db_id"]])
            sql['question'] = sql['question'].replace("  "," ")
            sql['question'] = sql['question'].replace("  "," ")

            sql['question'] = sql['question'][0].upper() + sql['question'][1:]
    else:
        for i,sql in enumerate(sqls):
            if ONE_ID and i != ONE_ID:
                continue
            sql['question'] = sql['question'].replace('‘',"'")
            sql['question'] = sql['question'].replace('’',"'")
            sql['question'] = sql['question'].replace("”","\"")
            sql['question'] = sql['question'].replace("“","\"")
            sql['question'] = sql['question'].replace("“","\"")
            sql['question'] = sql['question'].replace("`","'")
            sql['question'] = sql['question'].replace("（","(")
            sql['question'] = sql['question'].replace("）",")")
            sql['question'] = sql['question'].replace('('," ( ")
            sql['question'] = sql['question'].replace(')'," ) ")
            sql['question'] = sql['question'].replace('"',"'")
            sql['question'] = sql['question'].replace('  '," ")
            sql['question'] = sql['question'].replace('  '," ")
            sql['question'] = sql['question'].replace('  '," ")

            sentences = nlp(sql['question'])
            sql['or_question'] = [tok for tok in sentences]
            split_final = " ".join([tok.text for tok in sentences]).split(" ")[-1]
            if len(split_final) == 2 and split_final[0].isalpha() and not split_final[1].isalpha(): 
                sql['question'] = sql['question'][0:-1] + " " + sql['question'][-1]
    sqls2=[]
    for i,sql in enumerate(sqls):
        if ONE_ID and i!=ONE_ID:
            continue
        if ONE_ID and keep_original_question: 
            all_schema[sql["db_id"]] = Schema_Token(_tokenizer,None,table_dict[sql["db_id"]],None)
        sql['rid'] = i
        if keep_original_question:
            sql['question'] = sql['question'].replace("  "," ")
            sql['question'] = sql['question'].replace("  "," ")
        sentences = nlp(sql['question'])
        if keep_original_question:
            sql['question'] =  " ".join([tok.text for tok in sentences])
            sql['question'] = sql['question'][0].upper() + sql['question'][1:]

        sql['question_tag'] = [tok.tag_ for tok in sentences]
        sql['question_entt'] =  [tok.ent_type_ for tok in sentences]
        token_list = []

        _, table_match = return_table_name(TokenString(None,list(sentences)),all_schema[sql["db_id"]])
        offset = 0
        offset_list = []

        sentence_num = 0
        sentence_split = []
        for si, sent in enumerate(sentences.sents):
            token_list += sentence_cut(sent,table_match,offset,si,len(list(sentences.sents)))
            if token_list[-1][0].text in [".","?"]:
                sentence_num += 1
                sentence_split.append(len(token_list))
            offset += len(list(sent))
            offset_list.append(offset)
        token_list = re_analyse_sentence(token_list)
        token_list = re_analyse_sentence(token_list)
        token_list = anaylse_punctuate(token_list,sql['question'],all_schema[sql["db_id"]],all_word,keep_original_question)

        token_list = reset_uncontinue_type(token_list,sentence_num)
        token_list = correct_121_pattern(token_list,sentence_split)
        token_list = correct_121_pattern(token_list,sentence_split)
        token_list = correct_special_pattern(token_list,sql['question'],offset_list)
        token_list = reset_uncontinue_type(token_list,sentence_num)
        

        sql['table_match'] = table_match
        sql['question_dep'] = dump_language_feature(sentences)

        # analyse the second time: (Based on pattern)
        sql["db_match"] = None
        token_list,previous_col_match,previous_db_match,patterns = pattern_sentence_analyse(token_list,sql,all_schema[sql["db_id"]],sentence_num,1)
        token_list,previous_col_match,previous_db_match,patterns = pattern_sentence_analyse(token_list,sql,all_schema[sql["db_id"]],sentence_num,2,patterns)

        token_list = correct_121_pattern(token_list,sentence_split)
        # token_list = final_correct_special_pattern(token_list)
        col_match = []
        for key in previous_col_match:
            col_match.extend(key) 

        token_list = easy_cut(token_list,table_match,col_match)
        token_list = two_setence_analyse(list(sentences.sents),token_list)
        select_split(token_list,col_match,previous_db_match)
        
        token_list,offset,new_insert_idxs = pattern_generate_col(token_list,sql,all_schema[sql["db_id"]],previous_col_match,previous_db_match,use_pattern_generate_col,sentence_num)
        
        sql['db_match'] = []
        for key in previous_db_match.keys():
            sql['db_match'].extend(previous_db_match[key]) 
        
        
        if new_insert_idxs:
            sql['question'] = " ".join([tok[0].text for tok in token_list])
            sql['question_tag'] = [tok[0].tag_ for tok in token_list]
            sql['question_entt'] =  [tok[0].ent_type_ for tok in token_list]

            for idx in new_insert_idxs:
                sql['table_match'].insert(idx,[])
                sql['db_match'].insert(idx,[])
                col_match.insert(idx,[])
                sql['question_dep']["data"].insert(idx,[])
                for jj,root_i in enumerate(sql['question_dep']["root"]):
                    if root_i >= idx:
                        sql['question_dep']["root"][jj] += 1
                        print(sql['question'])
                for jj,data_i in enumerate(sql['question_dep']["data"]):
                    for jjj,d_list in enumerate(data_i):
                        if d_list["idx"] >= idx:
                            sql['question_dep']["data"][jj][jjj]["idx"] += 1
            
            remove_idx_list = []
            for ti,tok in enumerate(token_list):
                if tok[0].text[0] == "*" and tok[0].text[-1] == "*" and len(tok[0].text) >= 3:
                    if sql['table_match'][ti] or sql['db_match'][ti] or sql['question_entt'][ti]:
                        for idx in new_insert_idxs:
                            if token_list[idx][0].text == tok[0].text[1:-1]:
                                sql['table_match'][idx] = sql['table_match'][ti]
                                sql['db_match'][idx] = sql['db_match'][ti]
                                sql['question_entt'][idx] = sql['question_entt'][ti]
                    remove_idx_list.append(ti)
            for ri in reversed(remove_idx_list):
                del sql['table_match'][ri]
                del sql['db_match'][ri]
                del sql['question_entt'][ri]
                del sql['question_tag'][ri]
                del sql['question_dep']["data"][ri]
                del token_list[ri]
            token_list = reset_uncontinue_type(token_list)
            

        # if not keep_original_question:
        token_list,sql,col_match = db_correction(token_list,col_match,sql,all_schema[sql["db_id"]])
        sql = db_continue_correction(token_list, sql, all_schema[sql["db_id"]])

        token_list = final_cut(token_list,sql['db_match'],sql['table_match'],col_match,sql['question'],keep_original_question)
        token_list = final_correct_special_pattern(token_list)

        token_list = combine_none_subquestion(token_list,sql,all_schema[sql["db_id"]],sentence_num,col_match)
        token_list = combine_none_subquestion(token_list,sql,all_schema[sql["db_id"]],sentence_num,col_match)
        if token_list[0][1] != token_list[1][1]:
            token_list[0][1] = token_list[1][1]
        sql['question'] = " ".join([tok[0].text for tok in token_list])

        sql['question_type'] = reshap_token(token_list)
        assert len(sql['db_match']) == len(sql['question_type'])

        ########################################
        replace_old,replace_new = question_repair_select_part(sql,token_list,all_schema[sql["db_id"]])
        if replace_old:
            for old,new in zip(replace_old,replace_new):
                if sql['question'].count(" " + old+" ") == 1:
                    sql['question'] = sql['question'].replace(" " + old+" "," " + new+" ")
            sentences = nlp(sql['question'])
            sql['question_tag'] = [tok.tag_ for tok in sentences]
            sql['question_entt'] =  [tok.ent_type_ for tok in sentences]

        if 'query_toks' in sql.keys():
            sql.pop('query_toks')
        if 'question_toks' in sql.keys():
            sql.pop('question_toks')
        if 'or_question' in sql.keys():
            sql.pop('or_question')

        sql["full_db_match"] = generate_full_db_match(token_list,col_match,sql,all_schema[sql["db_id"]])

        if 'question_toks' not in sql.keys():
            sql['question_toks'] = sql['question'].split(" ")
        
        print(i)
        try:
            print(sql['question'])
        except:
            print("'ascii' codec can't encode character")
        if ONE_ID:
            print(sql['db_match'])
            print(previous_col_match)
            print(sql['table_match'])
        if token_list:
            print(sql['question_type'])
            print([str(rl)+" "+tok for rl,tok in zip(sql['question_type'],sql['question'].split(" "))])
    return sqls


ONE_ID = 0
def construct_hyper_param():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_file", default='NatSQLv1_6/dev.json', type=str)
    parser.add_argument("--table_file", default='NatSQLv1_6/tables.json', type=str)
    parser.add_argument("--out_file", default='dev-or.json', type=str)
    parser.add_argument('--keep_or', action='store_true', default=False)
    parser.add_argument("--use_pattern_generate_col", action='store_true', default=False)
    args = parser.parse_args()
    return args
    
if __name__ == '__main__':
    args = construct_hyper_param()
    all_word = pickle.load(open('data/20k-original.pkl', 'rb'))
    sqls = preprocess_sql(sql_path=args.in_file,table_path=args.table_file,all_word=all_word,keep_original_question = args.keep_or, use_pattern_generate_col=args.use_pattern_generate_col)
    if not ONE_ID:
        json.dump(sqls,open(args.out_file,'w'), indent=2)
        pass