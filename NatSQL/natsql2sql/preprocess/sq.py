import copy
from .utils import str_is_num
from .TokenString import SToken

class QuestionSQL():
    def __init__(self,sql,_tokenizer,tokens=None):
        self.question = ""
        self.question_type = []
        self.table_match = []
        self.sequence_tag = []
        self.sequence_dep = []
        self.sequence_entt = []
        tmp_sequence_dep = []
        for ssl,tm,ent,st,sd,i in zip(sql.sub_sequence_list, sql.table_match, sql.sequence_entt, sql.sequence_tag, sql.sequence_dep["data"], range(len(sql.sequence_tag))):
            self.question += ssl + " "
            for s_tm,s_entt,s_st,s_sd in zip(tm,ent,st,sd):
                self.question_type.append(sql.sub_sequence_type[i])
                self.table_match.append(s_tm)
                self.sequence_entt.append(s_entt)
                self.sequence_tag.append(s_st)
                tmp_sequence_dep.append(s_sd)
        self.question = self.question[:-1] 
        self.sequence_dep = {"root":sql.sequence_dep["root"],"data":tmp_sequence_dep}
        if tokens:
            self.question_tokens = tokens
        else:
            self.question_tokens = _tokenizer.tokenize(self.question)
            if len(self.question_tokens) != len(self.table_match):
                assert len(self.table_match) == self.question.count(" ")+1, "QuestionSQL Length Exception"
                j = 0
                for i,tok in enumerate(self.question.split(" ")):
                    if tok != self.question_tokens[j].text and tok.startswith(self.question_tokens[j].text):
                        text_t = self.question_tokens[j].text
                        for c in range(j+1,len(self.question_tokens),1):
                            text_t += self.question_tokens[c].text
                            if text_t == tok:
                                self.question_tokens[j] = SToken(text=text_t)
                                for cc in range(c,j,-1):
                                    del self.question_tokens[cc]
                                break
                    j += 1



class SubQuestion():
    def __init__(self,question,question_type,table_match,question_tag,question_dep,question_entt,sql,run_special_replace = True,db=None,sentence_num=1):
        
        db_match = sql["db_match"] if "db_match" in sql else None
        col_match = sql["col_match"] if "col_match" in sql else None
        pattern_tokens = sql["pattern_tok"] if "pattern_tok" in sql else None
        question_or = sql["question_or"] if "question_or" in sql else None
        full_db_match = sql["full_db_match"] if "full_db_match" in sql else None
        question_lemma = sql["question_lemma"] if "question_lemma" in sql else None
        
        if run_special_replace:
            col_match = None
            question = copy.deepcopy(question)
            question_type = copy.deepcopy(question_type)
            table_match = copy.deepcopy(table_match)
            question_tag = copy.deepcopy(question_tag)
            question_dep = copy.deepcopy(question_dep)
            question_entt = copy.deepcopy(question_entt)
            db_match = copy.deepcopy(db_match)
            question,question_type,table_match,question_tag,question_dep,toks_idxs,question_entt,db_match = self.special_replace(question,question_type,table_match,question_tag,question_dep,question_entt,db_match)
        
        self.sentence_num = sentence_num
        toks = question.split(" ")
        toks_idxs = [i for i in range(len(toks))]
        last_type = -1
        sub_sequence_list = []
        sub_sequence_type = []

        new_table_match = []
        sub_table_match = []
        new_table_match_weight = []
        sub_table_match_weight = []

        new_sequence_tag = []
        sub_sequence_tag = []

        new_sequence_dep = []
        sub_sequence_dep = []

        new_sequence_idx = []
        sub_sequence_idx = []

        new_sequence_entt = []
        sub_sequence_entt = []

        sentence_idx = 0
        sub_sentence_idx = []

        if db_match:
            db_match2 = db_match
        else:
            db_match2 = [[]] * len(question_entt)
        if not full_db_match:
            full_db_match = db_match2
        new_db_match = []
        sub_db_match = []
        new_full_db_match = []
        sub_full_db_match = []

        if col_match:
            col_match2 = col_match
        else:
            col_match2 = [[]] * len(question_entt)
        new_col_match = []
        sub_col_match = []

        if not pattern_tokens:
            pattern_tokens = [[]] * len(question_entt)
        new_pattern_token = []
        sub_pattern_token = []

        
        sub_sequence = ""
        last_tok = None
        for tok,type_,tm,qt,qd,idx,qe,db_m,col_m,pt,f_db_m in zip(toks,question_type,table_match,question_tag,question_dep["data"],toks_idxs,question_entt,db_match2,col_match2,pattern_tokens,full_db_match):
            if type_ != last_type or (last_tok in [".","?"] and type_ <= 1):
                if sub_sequence:
                    sub_sequence_list.append(sub_sequence)
                    sub_sequence_type.append(last_type)
                    new_table_match.append(sub_table_match)
                    new_table_match_weight.append(sub_table_match_weight)
                    new_sequence_tag.append(sub_sequence_tag)
                    new_sequence_dep.append(sub_sequence_dep)
                    new_sequence_idx.append(sub_sequence_idx)
                    new_sequence_entt.append(sub_sequence_entt)
                    sub_table_match = []
                    sub_table_match_weight = []
                    sub_sequence_tag = []
                    sub_sequence_dep = []
                    sub_sequence_idx = []
                    sub_sequence_entt = []
                    if db_match:
                        new_db_match.append(sub_db_match)
                        sub_db_match = []
                    new_full_db_match.append(sub_full_db_match)
                    sub_full_db_match = []
                    if col_match:
                        new_col_match.append(sub_col_match)
                        sub_col_match = []
                    new_pattern_token.append(sub_pattern_token)
                    sub_pattern_token = []

                    sub_sentence_idx.append(sentence_idx)
                    if last_tok in [".","?"]:
                        sentence_idx += 1
                sub_sequence = tok
                last_type = type_
            else:
                sub_sequence += " " + tok

            # There are two table match. OR is the table match from preprocess_sql.py while others are from word_....
            if not tm:
                sub_table_match.append(tm)
                sub_table_match_weight.append(tm)
            elif self.original_match_type(tm):
                sub_table_match.append(list(set(tm[0])))
                sub_table_match_weight.append([0]*len(sub_table_match))
            else:
                # sub_table_match.append(list(set([tm_[0] for tm_ in tm])) if tm else [])
                sub_table_match.append([])
                sub_table_match_weight.append([])
                for tm_ in tm:
                    if tm_ and tm_[0] not in sub_table_match[-1]:
                        sub_table_match[-1].append(tm_[0])
                        sub_table_match_weight[-1].append(tm_[1])
                    elif sub_table_match_weight[-1][sub_table_match[-1].index(tm_[0])] < tm_[1]:
                        sub_table_match_weight[-1][sub_table_match[-1].index(tm_[0])] = tm_[1]
            sub_sequence_tag.append(qt)
            sub_sequence_dep.append(qd)
            sub_sequence_idx.append(idx)
            sub_sequence_entt.append(qe)
            sub_db_match.append(db_m)
            sub_full_db_match.append(f_db_m)
            sub_col_match.append(col_m)
            sub_pattern_token.append(pt)


            last_tok = tok
        new_table_match.append(sub_table_match)
        new_table_match_weight.append(sub_table_match_weight)
        sub_sequence_list.append(sub_sequence)
        sub_sequence_type.append(last_type)
        sub_sentence_idx.append(sentence_idx)
        new_sequence_tag.append(sub_sequence_tag)
        new_sequence_dep.append(sub_sequence_dep)
        new_sequence_idx.append(sub_sequence_idx)
        new_sequence_entt.append(sub_sequence_entt)
        self.sub_sequence_list = sub_sequence_list
        self.sub_sequence_type = sub_sequence_type
        self.sub_sentence_idx = sub_sentence_idx
        self.table_match = new_table_match
        self.table_match_weight = new_table_match_weight
        self.sequence_tag = new_sequence_tag
        self.sequence_dep = {"root":question_dep["root"],"data":new_sequence_dep}
        self.sequence_entt = new_sequence_entt
        if db_match:
            new_db_match.append(sub_db_match)
            self.db_match = new_db_match
        else:
            self.db_match = [None] * len(self.table_match)
        new_full_db_match.append(sub_full_db_match)
        self.full_db_match = new_full_db_match
        if col_match:
            new_col_match.append(sub_col_match)
            self.col_match = new_col_match
        else:
            self.col_match = [[]]*len(new_sequence_entt)
        if pattern_tokens and pattern_tokens[0]:
            new_pattern_token.append(sub_pattern_token)
            self.pattern_tok = new_pattern_token
        else:
            self.pattern_tok = [[]]*len(new_sequence_entt)

        self.original_idx = []
        self.idx2sub_id = []
        idx = 0
        for nst_i,nst in enumerate(new_sequence_tag):
            sub_sequence_idx = []
            for t_i,t in enumerate(nst):
                sub_sequence_idx.append(idx)
                self.idx2sub_id.append([nst_i,t_i])
                idx += 1
            self.original_idx.append(sub_sequence_idx)
        self.offset = []
        for j,idxs in enumerate(self.original_idx):
            self.offset.append(idxs[0])
        self.question_or = question_or
        if not question_or:
            self.question_or = question.lower()
        if question_lemma:
            self.question_lemma = question_lemma
        else:
            self.question_lemma = self.question_or
        self.sub_sequence_toks = [sl.split(" ") for sl in self.sub_sequence_list]
        lemma_toks = self.question_lemma.split(" ")
        self.sub_sequence_lemma_toks = []
        for oi in self.original_idx:
            self.sub_sequence_lemma_toks.append([lemma_toks[i] for i in oi])

    def clean_data(self):
        self.sub_sequence_list = []
        self.sub_sequence_type = []
        self.sub_sentence_idx = []
        self.table_match = []
        self.table_match_weight = []
        self.sequence_tag = []
        self.sequence_dep = None
        self.sequence_entt = []
        self.db_match = []
        self.full_db_match = []
        self.col_match = []
        self.pattern_tok = []
        self.original_idx = []
        self.idx2sub_id = []
        self.offset = []
        self.question_or = None
        self.question_lemma = None
        self.sub_sequence_toks = []
        self.sub_sequence_lemma_toks = []

    def add_sub_element(self, sq, sq_idx , insert_idx = None):
        if insert_idx == None:
            insert_idx = len(self.sub_sequence_list)
        self.sub_sequence_list.insert(insert_idx,sq.sub_sequence_list[sq_idx])
        self.sub_sequence_type.insert(insert_idx,sq.sub_sequence_type[sq_idx])
        self.sub_sentence_idx.insert(insert_idx,sq.sub_sentence_idx[sq_idx])
        self.table_match.insert(insert_idx,sq.table_match[sq_idx])
        self.table_match_weight.insert(insert_idx,sq.table_match_weight[sq_idx])
        self.sequence_tag.insert(insert_idx,sq.sequence_tag[sq_idx])
        self.sequence_entt.insert(insert_idx,sq.sequence_entt[sq_idx])
        self.db_match.insert(insert_idx,sq.db_match[sq_idx])
        self.full_db_match.insert(insert_idx,sq.full_db_match[sq_idx])
        self.col_match.insert(insert_idx,sq.col_match[sq_idx])
        self.pattern_tok.insert(insert_idx,sq.pattern_tok[sq_idx])
        self.sub_sequence_toks.insert(insert_idx,sq.sub_sequence_toks[sq_idx])
        self.sub_sequence_lemma_toks.insert(insert_idx,sq.sub_sequence_lemma_toks[sq_idx])        
        




    def original_match_type(self, tm):
        if len(tm) == 4:
            if len(tm[0]) == 2:
                if type(tm[0][1]) == float or type(tm[1][1]) == float or type(tm[2][1]) == float or type(tm[3][1]) == float:
                    return False
                elif tm[0] == tm[3] or tm[1] == tm[2] or tm[0] == tm[2]:
                    return True
                else:
                    return False
            return True
        return False
        
    
    def sentence_combine(self,combine_type=0,with_token_pattern=False,type_offset=0):
        re_quest_list = []
        re_table_list = []
        re_list_idx = []
        re_db_list = []
        pattern_tok = []
        if combine_type == 0:
            for sub_seq,sub_type,sub_table,idx,db_m,pt in zip(self.sub_sequence_list, self.sub_sequence_type, self.table_match, range(len(self.sub_sequence_list)),self.db_match,self.pattern_tok):
                if sub_type+type_offset in [0,1] and len(self.original_idx[idx])>=1:
                    re_quest_list.append(sub_seq)
                    re_table_list.append(sub_table)
                    re_list_idx.append(idx)
                    re_db_list.append(db_m)
                    pattern_tok.append(pt)
        else:
            for sub_seq,sub_type,sub_table,idx,db_m,pt in zip(self.sub_sequence_list, self.sub_sequence_type, self.table_match, range(len(self.sub_sequence_list)),self.db_match,self.pattern_tok):
                if sub_type+type_offset > 1 and len(self.original_idx[idx])>=1:
                    re_quest_list.append(sub_seq)
                    re_table_list.append(sub_table)
                    re_list_idx.append(idx)
                    re_db_list.append(db_m)
                    pattern_tok.append(pt)

        if with_token_pattern:
            return re_quest_list, re_table_list, re_list_idx, re_db_list, pattern_tok
        return re_quest_list, re_table_list, re_list_idx, re_db_list


    def table_match_index(self, list_idx, schema, recurrent=True):


        re_match_idx = []
        if list_idx >= len(self.table_match) or list_idx < 0:
            return [-1]
        else:
            for t in self.table_match[list_idx]:
                if t:
                    re_match_idx.extend(t)
            if list_idx > 0 and recurrent:
                for i in range(list_idx,0,-1):
                    if len(self.question_tokens[i]) > 1 and (self.question_tokens[i][0].lower_ in ["that","who","whose","what","where","while","whom","which","and","or"] or self.question_tokens[i][1].lower_ in ["that","who","whose","what","where","while","whom","which","and","or"]):
                        if self.question_tokens[i][0].lower_ in ["that","who","whose","what","where","while","whom","which"] or self.question_tokens[i][1].lower_ in ["that","who","whose","what","where","while","whom","which"] :
                            if len(self.table_match[i-1]) >= 2 and list_idx == i:
                                re_match_idx.extend(self.table_match[i-1][-1])
                                re_match_idx.extend(self.table_match[i-1][-2])
                                if self.question_tokens[i-1][-1].lower_ in ["and","or"] and i > 1 and len(self.table_match[i-2]) > 2:
                                    re_match_idx.extend(self.table_match[i-2][-1])
                                    re_match_idx.extend(self.table_match[i-2][-2])
                            if self.db_match and self.db_match[i-1] and self.db_match[i-1][-1]:
                                for db in self.db_match[i-1][-1]:
                                    re_match_idx.append(schema.column_names_original[db[0]][0])
                        break
                    else:
                        question_str = self.sub_sequence_list[i].lower()
                        if  len(self.question_tokens[i]) > 1 and ("that " in question_str or "who " in question_str or "whose"in question_str or "what " in question_str or "where " in question_str or "while " in question_str or "whom " in question_str or "which " in question_str or self.question_tokens[i][0].lower_ in ["and","or"] or self.question_tokens[i][1].lower_ in ["and","or"]):
                            pass
                        else:
                            re_match_idx.extend(self.table_match_index(i-1, schema,False))
            
            if not re_match_idx:
                if self.col_match[list_idx]:
                    for cm in self.col_match[list_idx]:
                        if cm:
                            for ci,c in enumerate(cm[0]):
                                if cm[1][ci] > 1 or cm[2][ci] == 1:
                                    re_match_idx.append(schema.column_names_original[c][0])
                if self.db_match[list_idx]:
                    for dbm in self.db_match[list_idx]:
                        for cdb in dbm:
                            re_match_idx.append(schema.column_names_original[cdb[0]][0])
            return list(set(re_match_idx))


    def get_select_table_index(self, list_idx, select_type, schema):
        if self.sub_sequence_type[list_idx] <= select_type:
            select_num = 0
            for i in self.sub_sequence_type:
                if i <= select_type:
                    select_num += 1
            if select_num == 2:
                for i,type_ in enumerate(self.sub_sequence_type):
                    if type_ <= select_type and i != list_idx:
                        if self.question_tokens[i][0].lower_ == "which" and self.table_match[i].count([]) < len(self.table_match[i]):
                            return self.table_match_index(i, schema)
                    elif type_ <= select_type and i == list_idx and i > 0 and self.sub_sequence_type[0] <= select_type:
                        if not self.question_tokens[i][0].text.islower() and self.table_match[0].count([]) + 1 == len(self.table_match[0]):
                            return self.table_match_index(0, schema)
        return self.table_match_index(list_idx, schema)

    

    def tokenize(self,qsql:QuestionSQL):
        self.question_tokens = [[] for i in self.sub_sequence_list]
        for i, idxs in enumerate(self.original_idx):
            for idx in idxs:
                self.question_tokens[i].append(qsql.question_tokens[idx])


    def gennerate_col_match(self,schema):
        col_match = []
        for cols in self.col_match:
            for col in cols:
                cc = []
                cc_separate_1 = []
                cc_separate_2 = []
                if col:
                    for c,l,t in zip(col[0],col[1],col[2]):
                        if t == 1:
                            if c in cc_separate_1:
                                if 1 > cc_separate_2[cc_separate_1.index(c)]:
                                    cc[cc_separate_1.index(c)] = (c, 1)
                                    cc_separate_2[cc_separate_1.index(c)] = (c, 1)
                            else:
                                cc.append((c,1))
                                cc_separate_1.append(c)
                                cc_separate_2.append(1)
                        else:
                            if " | " not in schema.column_tokens_lemma_str[c]: 
                                value = round(l/len(schema.column_tokens[c]),2)
                                if c in cc_separate_1:
                                    if value > cc_separate_2[cc_separate_1.index(c)]:
                                        cc[cc_separate_1.index(c)] = (c, value)
                                        cc_separate_2[cc_separate_1.index(c)] = (c, value)
                                else:
                                    cc.append((c, value))
                                    cc_separate_1.append(c)
                                    cc_separate_2.append(value)
                            else:
                                all_col_len = []
                                for col in schema.column_tokens_lemma_str[c].split(" | "):
                                    all_col_len.append(col.count(" ")+1)
                                value = round(l/max(all_col_len),2)
                                if c in cc_separate_1:
                                    if value > cc_separate_2[cc_separate_1.index(c)]:
                                        cc[cc_separate_1.index(c)] = (c, value)
                                        cc_separate_2[cc_separate_1.index(c)] = (c, value)
                                else:
                                    cc.append((c, value))
                                    cc_separate_1.append(c)
                                    cc_separate_2.append(value)
                col_match.append(cc)
        return col_match
    
    def gennerate_db_match(self):
        db_match = []
        for dbs in self.db_match:
            for db in dbs:
                cc = []
                if db:
                    for cdb in db:
                        cc.append(cdb[0])
                db_match.append(cc)
        return db_match


    def gennerate_table_match(self,schema,table_match):
        col_match = []
        for q_idx,col in enumerate(table_match):
            cc = []
            cc_separate_1 = []
            cc_separate_2 = []
            if col:
                for c,l,t in zip(col[0],col[1],col[2]):
                    if t == 1:
                        if c in cc_separate_1:
                            if 1 > cc_separate_2[cc_separate_1.index(c)]:
                                cc[cc_separate_1.index(c)] = (c, 1)
                                cc_separate_2[cc_separate_1.index(c)] = (c, 1)
                        else:
                            cc.append((c,1))
                            cc_separate_1.append(c)
                            cc_separate_2.append(1)
                    else:
                        if " | " not in schema.table_tokens_lemma_str[c]: 
                            value = round(l/len(schema.table_tokens[c]),2)
                            if c in cc_separate_1:
                                if value > cc_separate_2[cc_separate_1.index(c)]:
                                    cc[cc_separate_1.index(c)] = (c, value)
                                    cc_separate_2[cc_separate_1.index(c)] = (c, value)
                            else:
                                cc.append((c, value))
                                cc_separate_1.append(c)
                                cc_separate_2.append(value)
                        else:
                            all_col_len = []
                            if l == 1:
                                for ccol in schema.table_tokens_lemma_str[c].split(" | "):
                                    if self.question_tokens[self.idx2sub_id[q_idx][0]][self.idx2sub_id[q_idx][1]].lemma_ + " " in ccol:
                                        all_col_len.append(ccol.count(" ")+1)
                                    elif " " + self.question_tokens[self.idx2sub_id[q_idx][0]][self.idx2sub_id[q_idx][1]].lemma_ in ccol:
                                        all_col_len.append(ccol.count(" ")+1)
                            if not all_col_len:
                                for ccol in schema.table_tokens_lemma_str[c].split(" | "):
                                    all_col_len.append(ccol.count(" ")+1)
                            value = round(l/max(all_col_len),2)
                            if c in cc_separate_1:
                                if value > cc_separate_2[cc_separate_1.index(c)]:
                                    cc[cc_separate_1.index(c)] = (c, value)
                                    cc_separate_2[cc_separate_1.index(c)] = (c, value)
                            else:
                                cc.append((c,value))
                                cc_separate_1.append(c)
                                cc_separate_2.append(value)
                        
            col_match.append(cc)
        return col_match

    
    def gennerate_original_matchs(self):
        db_match = []
        col_match = []
        for dbs in self.db_match:
            for db in dbs:
                db_match.append(db)
        for dbs in self.col_match:
            for db in dbs:
                col_match.append(db)

        return db_match,col_match
        


    def gennerate_question(self,use_token=True):
        question = ""
        if use_token:
            for ql in self.question_tokens:
                for tok in ql:
                    question += tok.text + " "
        else:
            for sql in self.sub_sequence_list:
                question += sql + " "
        return question[:-1]
        

    def gennerate_pattern_tok(self):
        tokens = []
        for ql in self.pattern_tok:
            tokens.extend(ql)
        return tokens

    def gennerate_question_lemma(self):
        question = ""
        for ql in self.question_tokens:
            for tok in ql:
                if tok.lemma_ == '-PRON-':
                    question += tok.text + " "
                else:
                    question += tok.lemma_ + " "
        return question[:-1]

    def special_replace(self,question,question_type,table_match,question_tag,question_dep,question_entt,db_match):
        """
        replace to special token with predefined token, such as from how many to number of
        """

        def prepare_to_delete_node(idx,question_dep):
            if idx in question_dep["root"]:
                data_in_node = []
                for d in question_dep["data"][idx]:
                    data_in_node.append(d)
                for i,dep in enumerate(question_dep["data"]):
                    if i == data_in_node[0]["idx"]: 
                        dep.extend(data_in_node)
                        for j,d in enumerate(dep):
                            if d["idx"] == i:
                                del dep[j]
                        for k in range(len(question_dep["root"])):
                            if question_dep["root"][k] == idx:
                                question_dep["root"][k] = i
                                return question_dep
            else:
                data_in_node = []
                for d in question_dep["data"][idx]:
                    data_in_node.append(d)
                for dep in question_dep["data"]:
                    for d in dep:
                        if d['idx'] == idx: 
                            dep.extend(data_in_node)
                            return question_dep
            return question_dep

        def delete_question_dep(idx,question_dep):
            for j, dep in enumerate(question_dep["data"]):
                if j == idx:
                    question_dep = prepare_to_delete_node(idx,question_dep)
                    del question_dep["data"][idx]
                    break
            for j, dep in enumerate(question_dep["data"]):
                for i,d in enumerate(dep) :
                    if d['idx'] == idx:
                        del dep[i]
                        break
            for j, dep in enumerate(question_dep["data"]):
                for i,d in enumerate(dep) :
                    if d['idx'] > idx:
                        d['idx'] -= 1
            return question_dep

        
        
        def remove(check, remove_tok_idxs, qtoks, question_type, table_match,question_tag,question_dep,toks_idxs,question_entt,db_match):
            """
            !!!The order of remove_tok_idxs must from high to low.!!!
            """
            len_old = len(check)
            for i in range(len(qtoks)-len_old+1):
                match = [False] * len_old
                for j in range(len_old):
                    if qtoks[i+j] == check[j]:
                        match[j] = True
                if False not in match:
                    for id_ in remove_tok_idxs:
                        del qtoks[id_+i]
                        del question_type[id_+i]
                        del table_match[id_+i]
                        del question_tag[id_+i]
                        question_dep = delete_question_dep(id_+i,question_dep)
                        del toks_idxs[id_+i]
                        del question_entt[id_+i]
                        if db_match:
                            del db_match[id_+i]
                    return qtoks,question_type,table_match
            return qtoks,question_type,table_match,question_tag,question_dep,toks_idxs,question_entt, db_match

        # question = question.replace("many singers",", and")
        toks_idxs = [i for i in range(len(question_tag))]
        qtoks = question.split(" ")
        while(True):
            if " average number of " in question:
                remove(["average","number","of"], [2,1], qtoks, question_type, table_match,question_tag,question_dep,toks_idxs,question_entt,db_match)
                question = " ".join(qtoks)
            else:
                break
        while(True):
            if " , and " in question:
                remove([",","and"], [0], qtoks, question_type, table_match,question_tag,question_dep,toks_idxs,question_entt,db_match)
                question = " ".join(qtoks)
            else:
                break

        question = question.lower().replace("how many ","number of ")
        return question,question_type,table_match,question_tag,question_dep,toks_idxs,question_entt ,db_match
