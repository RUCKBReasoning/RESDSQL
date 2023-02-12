import sqlite3,os
import re
from functools import lru_cache
from config import DATABASE_PATH
from .utils import str_is_num,str_is_special_num,get_punctuation_word
from .Schema_Token import Schema_Token
from .TokenString import lemmatization
from .match import ALL_SPECIAL_WORD
from .utils import str_is_num

class DBEngine:
    DB_SHARE = None
    def __init__(self, schema):
        if schema and type(schema) == Schema_Token:
            fdb = schema.db_id
            self.table_list = schema.table_names_original
            self.column_list = schema.column_names_original
            self.column_name_list = schema.column_tokens_text_str
            self.column_types = schema.column_types

        elif schema and type(schema) == dict:
            fdb = schema["db_id"]
            self.table_list = schema["table_names_original"]
            self.column_list = schema["column_names_original"]
            self.column_name_list = [col[1] for col in schema["column_names"]]
            self.column_types = schema["column_types"]

        else:
            return
        self.db_id = fdb
        file_path = os.path.join(DATABASE_PATH(),fdb,fdb+ ".sqlite")
        try:
            self.db = sqlite3.connect(file_path)
        except Exception as e:
            try:
                self.db = sqlite3.connect(file_path)
            except Exception as e:
                raise Exception(f"Can't connect to SQL: {e} in path {file_path}")
    
    def close(self):
        try:
            self.db.close()
        except:
            pass

    def col_data_samples(self,table_idx):
        all_cols = []
        for col,n_col,c_type in zip(self.column_list,self.column_name_list,self.column_types):
            if col[0] != table_idx:
                continue
            query = "select distinct "+col[1]+" from " + self.table_list[table_idx] + " order by "+col[1]+" limit 5"
            cursor = self.db.cursor()
            try:
                self.db.text_factory = str
                cursor.execute(query)
                values = cursor.fetchall()
                all_cols.append([v[0] for v in values if v[0] != "" ] )
            except:
                self.db.text_factory = lambda x: str(x, 'latin1')
                all_cols.append([])
                continue
        
        return all_cols

    def db_col_type_check(self,table_idx):
        def str_is_date(s):
            s = s.lower()
            if "a" in s or "e" in s or "o" in s or "u" in s:
                s = s.replace("-jan-","-1-")
                s = s.replace("-feb-","-2-")
                s = s.replace("-mar-","-3-")
                s = s.replace("-apr-","-4-")
                s = s.replace("-may-",'-5-')
                s = s.replace("-jun-","-6-")
                s = s.replace("-jul-","-7-")
                s = s.replace("-aug-","-8-")
                s = s.replace("-sept-","-9-")
                s = s.replace("-sep-","-9-")
                s = s.replace("-oct-","-10-")
                s = s.replace("-nov-","-11-")
                s = s.replace("-dec-","-12-")

            if re.fullmatch(r"^([1][5-9]\d{2}|[2][0]\d{2})$",s, flags=0):
                return "YEAR"
            elif str_is_num(s):
                return "NUM"
            elif re.fullmatch(r'((\d{4}((_|-|/){1}\d{1,2}){2})|(\d{1,2}(_|-|/)){2}(\d{4}|\d{2})){0,1}\s{0,1}(\d{2}(:\d{2}){1,2}){0,1}',s, flags=0):
                return "DATE"
            elif re.fullmatch(r'(\d{1,2}(st|nd|rd|th){0,1}(,|\s|-)){0,1}((J|j)(an|AN)(uary){0,1}|(F|f)(eb|EB)(ruary){0,1}|(M|m)(ar|AR)(ch){0,1}|(A|a)(pr|PR)(il){0,1}|(M|m)(ay|AY)|(J|j)(un|UN)(e){0,1}|(J|j)(ul|UL)(y){0,1}|(A|a)(ug|UG)(ust){0,1}|(S|s)(ep|EP)(tember){0,1}|(O|o)(ct|CT)(ober){0,1}|(N|n)(ov|OV)(ember){0,1}|(D|d)(ec|EC)(ember){0,1})(\s|,|-)(\d{1,2}(st|nd|rd|th){0,1}(\s|,){1,3}){0,1}(\d{4}|\d{2})',s, flags=0):
                return "DATE"
            if re.fullmatch(r"^(\d{1,2}:\d{1,2}(.){0,1}\d{0,2}(.){0,1}\d{0,2})$",s, flags=0):
                return "DATE"
            return None

        def str_is_year(values):
            for v in values:
                if v[0].strip() != "":
                    if not re.fullmatch(r"^[1-2]\d{3}$",v[0], flags=0):
                        return False

            return True

        
        bool_col = []
        all_cols = []
        for col,n_col,c_type in zip(self.column_list,self.column_name_list,self.column_types):
            if col[0] != table_idx:
                continue
            query = "select distinct "+col[1]+" from " + self.table_list[table_idx] + " order by "+col[1]+" limit 500"
            
            cursor = self.db.cursor()
            try:
                self.db.text_factory = str
                cursor.execute(query)
                values = cursor.fetchall()
            except:
                self.db.text_factory = lambda x: str(x, 'latin1')
                all_cols.append((col,-1))
                continue
            if n_col.lower() == "time":
                print(" ")
            if len(values) == 0 or (len(values) == 1 and values[0][0] in ["",None]):
                if n_col.lower().endswith(" date") or n_col.lower() == "data" or n_col.lower().endswith(" time") or n_col.lower() == "time" or n_col.lower() == "duration":
                    all_cols.append((col,3))
                elif n_col.lower().endswith(" year") or n_col.lower() == "year":
                    all_cols.append((col,4))
                else:
                    all_cols.append((col,-1))
                continue
            if len(values) == 2 or (len(values) == 3 and str(values[0][0]) == ""):
                if len(values) == 2:
                    left = str(values[0][0]).lower()
                    right = str(values[1][0]).lower()
                else:
                    left = str(values[1][0]).lower()
                    right = str(values[2][0]).lower()
                if (left == "0" and right == "1") or \
                   (left == "f" and right == "t") or (left == "false" and right == "true") or \
                   (left == "n" and right == "y") or (left == "no" and right == "yes") or \
                   (left == "f" and right == "m") or (left == "female" and right == "male") or \
                   (left == "f" and right == "s") or (left == "fail" and right == "success") or \
                   (left == "b" and right == "g") or (left == "bad" and right == "good") or \
                   (left == "cancelled" and right == "completed") or \
                   (left == "l" and right == "r") or (left == "left" and right == "right"):
                    bool_col.append(col)
                    all_cols.append((col,0))
                    continue
            is_str = False
            skip_once = True
            for v in values:
                if v[0] and v[0] not in ["", " ", "  ",  None, 'None','NONE','none', 'nil','NULL','Null','null','inf','-inf','+inf'] and type(v[0])==str and v[0][0].isalpha():
                    if  skip_once and len(values) > 7:
                        skip_once = False
                        continue
                    all_cols.append((col,1))
                    is_str = True
                    break
            if not is_str:
                skip_once = True
                check_max = 100
                for v in values:
                    if v[0] in ["", " ", "  ",  None, 'None','NONE','none', 'nil','NULL','Null','null','inf','-inf','+inf']:
                        continue
                    elif (v[0] or v[0] == 0) and  (type(v[0]) in [int,float] or ( type(v[0])==str and (str_is_date(v[0]) or str_is_num(v[0])) ) ):
                        check_max -= 1
                        if check_max <= 0:
                            break
                    else:
                        if  skip_once and len(values) > 7 and not v[0][0].isdigit():
                            skip_once = False
                            continue
                        all_cols.append((col,1))
                        is_str = True
                        break
                if is_str:
                    pass
                elif check_max < 100:
                    if len(values) >= 2:
                        n_col  = n_col.lower()
                        if v[0] == "":
                            all_cols.append((col,2))
                        elif type(v[0]) in [int,float] and (("year" == n_col) or ("year " in n_col) or n_col.lower().endswith(" year") or ("founded" == n_col)):
                            all_cols.append((col,4))
                        elif type(v[0]) in [int] and (n_col.lower().endswith(" date") or n_col.lower() == "data" or n_col.lower().endswith(" time") or n_col.lower() == "time" or n_col.lower() == "duration"):
                            all_cols.append((col,3))
                        elif type(v[0])==str:
                            ttt = str_is_date(v[0])
                            if ttt in ["DATE","TIME"]:#,"YEAR"
                                all_cols.append((col,3))
                            elif ttt == "YEAR" and (("year" == n_col) or ("year " in n_col) or (" year" in n_col) or ("founded" == n_col)):
                                all_cols.append((col,4))
                            elif ttt == "YEAR" and str_is_year(values):
                                all_cols.append((col,4))
                            elif ttt == "NUM"  and (n_col.lower().endswith(" date") or n_col.lower() == "data" or n_col.lower().endswith(" time") or n_col.lower() == "time" or n_col.lower() == "duration"):
                                all_cols.append((col,3))
                            else:
                                all_cols.append((col,2))
                        else:
                            all_cols.append((col,2))
                    else:
                        all_cols.append((col,-1))
                else:
                    all_cols.append((col,1))

        return bool_col,all_cols


    def db_content_are_same(self,col_1,col_2):
        query  = "select distinct "+self.column_list[col_1][1]+" from " + self.table_list[self.column_list[col_1][0]] +" limit 500"
        query2 = "select distinct "+self.column_list[col_2][1]+" from " + self.table_list[self.column_list[col_2][0]] +" limit 500"

        cursor = self.db.cursor()
        try:
            self.db.text_factory = str
            cursor.execute(query)
            values = cursor.fetchall()
            cursor.execute(query2)
            values2 = cursor.fetchall()
        except:
            self.db.text_factory = lambda x: str(x, 'latin1')
            all_cols.append((col,-1))
            return False
        if len(values) == 0 or len(values2) == 0:
            return True
        values,values2 = (values,values2) if len(values) > len(values2) else (values2,values)
        pass_one = False
        for v in values2:
            if v and v not in values:
                if pass_one:
                    return False
                else:
                    pass_one = True
        return True



    def contain_token(self, token, table, table_idx,all_utter_tokens, token_idx):
        where_condition = ""
        col_idx = []
        for i, col in enumerate(self.column_list):
            if col[0] == table_idx and col[1].strip() != "*":
                where_condition += "trim([" + col[1] + "]) like '" + token + "%' or "
                col_idx.append(i)
            if col[0] > table_idx:
                break
        if where_condition.endswith(" or "):
            where_condition = where_condition[:len(where_condition)-4]
            where_condition = " where " + where_condition
        query = "select distinct * from " + table + where_condition + " limit 500"
        
        cursor = self.db.cursor()
        try:
            self.db.text_factory = str
            cursor.execute(query)
            values = cursor.fetchall()
        except:
            self.db.text_factory = lambda x: str(x, 'latin1')
            return None,None
        return self.contain_token_column_idx(values,token,col_idx,all_utter_tokens,token_idx)

    def contain_exact_token(self, token, table, table_idx,all_utter_tokens, token_idx):
        where_condition = ""
        col_idx = []
        for i, col in enumerate(self.column_list):
            if col[0] == table_idx and col[1].strip() != "*":
                where_condition += "trim([" + col[1] + "]) = '" + token + "' or "
                col_idx.append(i)
            if col[0] > table_idx:
                break
        if where_condition.endswith(" or "):
            where_condition = where_condition[:len(where_condition)-4]
            where_condition = " where " + where_condition
        query = "select distinct * from " + table + where_condition + " limit 500"
        
        cursor = self.db.cursor()
        try:
            self.db.text_factory = str
            cursor.execute(query)
            values = cursor.fetchall()
        except:
            self.db.text_factory = lambda x: str(x, 'latin1')
            return None,None
        return self.contain_token_column_idx(values,token,col_idx,all_utter_tokens,token_idx)

    def query_return_match(self, str_return, one_token, all_utter_tokens,token_idx):
        db_string = str(str_return).strip()
        str_return = db_string.lower()
        one_token = one_token.lower()
        if str_return == one_token:
            return [token_idx,token_idx],db_string
        elif str_return and str_return.isalpha() and " " not in str_return and lemmatization(str_return) == one_token and one_token not in ALL_SPECIAL_WORD:
            return [token_idx,token_idx],db_string
        start_token_idx = token_idx
        token_idx += 1
        if all_utter_tokens:
            while len(str_return) > len(one_token) and token_idx < len(all_utter_tokens):
                next_token = one_token + " " + all_utter_tokens[token_idx].lower_
                if str_return == next_token or str_return+"s" == next_token or str_return == next_token+"s":
                    return [start_token_idx,token_idx],db_string
                elif str_return[:len(next_token)] != next_token:
                    next_token = one_token + all_utter_tokens[token_idx].lower_
                    if str_return[:len(next_token)] != next_token:
                        return None,None
                    elif str_return == next_token:
                        return [start_token_idx,token_idx],db_string
                    else:
                        one_token = next_token
                else:
                    one_token = next_token
                token_idx += 1
        return None,None


    def contain_token_column_idx(self,values,token,col_idx,all_utter_tokens,token_idx):
        if len(values) == 0:
            return [],None
        else:
            result = []
            final_string = ""
            loop = 5000
            for row in values:
                loop -= 1
                if loop <= 0:
                    break
                for i,v in enumerate(row) :
                    res,db_string = self.query_return_match(v, token, all_utter_tokens,token_idx)
                    res_final = [col_idx[i],res]
                    if res and res_final not in result:
                        result.append(res_final)
                        if len(db_string) > len(final_string):
                            final_string = db_string
            return result,final_string
        
    def get_db_structure_info(self):
        query = "select * from sqlite_master WHERE type=\"table\";"
        cursor = self.db.cursor()
        try:
            self.db.text_factory = str
            cursor.execute(query)
            values = cursor.fetchall()
        except:
            return None
        return values

    def check_disjoint_column(self,column_id):
        query = "select count(distinct "+str(self.column_list[column_id][1])+"),count("+str(self.column_list[column_id][1])+") from "+str(self.table_list[self.column_list[column_id][0]])+" ;"
        cursor = self.db.cursor()
        try:
            self.db.text_factory = str
            cursor.execute(query)
            values = cursor.fetchall()
            if values[0][0] != values[0][1]:
                return True
        except:
            return None
        return False

    @classmethod
    def new_db(cls, schema):
        if not (cls.DB_SHARE and cls.DB_SHARE.db_id == schema.db_id):
            if cls.DB_SHARE:
                cls.DB_SHARE.close()
            cls.DB_SHARE = DBEngine(schema)
        return cls.DB_SHARE


    def get_all_db_string(self):
        all_str = []
        for col,c_type in zip(self.column_list,self.column_types):
            if c_type != "text" or col[1] == "*":
                continue
            query = "select distinct "+col[1]+" from " + self.table_list[col[0]] + " limit 500"

            cursor = self.db.cursor()
            try:
                self.db.text_factory = str
                cursor.execute(query)
                values = cursor.fetchall()
            except:
                continue
            all_str.append(values)
        return all_str

def return_result(results):
    return_ = []
    col_list = []
    for r in results:
        if not return_:
            return_.append(r)
            col_list.append(r[0])
        else:
            not_add=False
            for i,r_ in enumerate(return_):
                if r_[1][1] < r[1][1]:
                    return_[i] = r
                    col_list[i] = r[0]
                elif r_[1][1] == r[1][1] and r[0] not in col_list:
                    not_add=True
            if not_add:
                col_list.append(r[0])
                return_.append(r)
    for r in range(len(col_list)-1,0,-1):
        if col_list.count(col_list[r]) > 1:
            del return_[r]
            del col_list[r]
    return return_

def datebase_match(schema,tok,tok_idx,utter_tokens,table_idx,cross_table=True):
    """

    return list:
        < [ col_in_table_index , [where_right_string_start_idx,where_right_string_stop_idx] ], ... >
    """

    def try_one_table(_db:DBEngine,tok,table,table_idx,utter_tokens,tok_idx):
        result,db_string = _db.contain_token(tok.text,table,table_idx, utter_tokens,tok_idx)
        if result:
            if len(result) == 1:
                return result
            return return_result(result)

        if not result and tok.text != tok.lemma_ and tok.tag_ not in ["JJS","JJR","RBR","RBS"]:
            result,db_string = _db.contain_token(tok.lemma_,table,table_idx,utter_tokens,tok_idx)
            if result:
                if len(result) == 1:
                    return result
                return return_result(result)
        return None


    _db = DBEngine.new_db(schema)
    match_col = []
    occupy_u_tokens = []
    if str_is_num(tok.text) and not str_is_special_num(tok.text) and (not tok.text.isalpha() or tok.text.islower()): 
        is_punt, punt_edge =  get_punctuation_word(utter_tokens,tok_idx,only_bool=False)
        if is_punt and (punt_edge[1] - punt_edge[0] > 1 or len(tok.text) > 2):
            pass
        else:
            return None
    if table_idx >= 0:
        result = try_one_table(_db,tok,schema.table_names_original[table_idx],table_idx,utter_tokens,tok_idx)
        if result:
            return result
    
    if cross_table:
        resultss = [ ]
        for j, table in enumerate(schema.table_names_original):
            if j == table_idx:
                continue
            result = try_one_table(_db,tok,schema.table_names_original[j],j,utter_tokens,tok_idx)
            if result:
                resultss.extend(result)
        return resultss
    return None


def datebase_match_tables(schema,tok,tok_idx,utter_tokens,table_idxs,return_all_match = False,search_extra_table=False):
    def match_tables(schema,tok,tok_idx,utter_tokens,table_idxs):
        all_ = []
        for t in table_idxs:
            result = datebase_match(schema,tok,tok_idx,utter_tokens,t,False)
            if result and (len(result) == 1 or return_all_match):
                all_.append(result)
            else:
                all_.append(None)
        return all_
    all_ = match_tables(schema,tok,tok_idx,utter_tokens,table_idxs)
    if not all_ or len(all_) == all_.count(None) or search_extra_table:
        all_2 = None
        new_table_idxs = []
        for j, table in enumerate(schema.table_names_original):
            if j not in table_idxs:
                new_table_idxs.append(j)
        if new_table_idxs:
            all_2 = match_tables(schema,tok,tok_idx,utter_tokens,new_table_idxs)
        if new_table_idxs and (not all_ or len(all_) == all_.count(None)):
            all_ = all_2
        elif search_extra_table and all_2:
            all_2 = [ a  for a in all_2 if a and a[0][1][1]>a[0][1][0] ]
            all_ = [ a2  for a in all_ if a for a2 in a ]
            max_all_ = max([ a[1][1]-a[1][0]  for a in all_ ])
            all_ = [all_]
            if all_2 and all_2[0][0][1][1] - all_2[0][0][1][0] > max_all_:
                all_ = all_2
    if return_all_match:
        for i in reversed(range(len(all_))):
            if not all_[i]:
                del all_[i]
        return all_
    elif all_.count(None) == len(all_) - 1:
        for r in all_:
            if r:
                return r
    return None

  
def get_match_col(db_match):
    if db_match:
        cols = set()
        for c in db_match:
            cols.add(c[0])
        return list(cols)
    return None



def get_database_string(schema, col_idx, all_tokens):
    """
        token: frist token, string
        col_idx: table name string
    """
    _db = DBEngine.new_db(schema)
    result,db_string = _db.contain_token(all_tokens[0].text,schema.table_names_original[schema.column_tokens_table_idx[col_idx]],schema.column_tokens_table_idx[col_idx], all_tokens,0)
    if not result:
        result,db_string = _db.contain_token(all_tokens[0].lemma_,schema.table_names_original[schema.column_tokens_table_idx[col_idx]],schema.column_tokens_table_idx[col_idx], all_tokens,0)
    return db_string



def get_database_col(schema, db_match, all_tokens):
    """
        token: frist token, string
        col_idx: table name string
    """
    _db = DBEngine.new_db(schema)
    cols = [c[0] for c in db_match]
    tbs  = [schema.column_tokens_table_idx[c] for c in cols]
    for i,t in enumerate(schema.table_names_original):
        if i in tbs:
            continue
        result,db_string = _db.contain_token(all_tokens[0].text,t,i, all_tokens,0)
        if not result:
            result,db_string = _db.contain_token(all_tokens[0].lemma_,t,i, all_tokens,0)
        if result:
            for r in result:
                assert r[0] not in cols
                cols.append(r[0])
    return cols
