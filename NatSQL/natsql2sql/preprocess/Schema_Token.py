from .match import AGG_WORDS,AGG_OPS,INFORMATION_WORDS,COUNTRYS
from .stemmer import MyStemmer

import copy

class Schema_Token():
    def __init__(self, _tokenizer, stemmer, table_dict, _concept_word):
        self.original_table = copy.deepcopy(table_dict)
        self.original_table["tc_fast"] = []
        if "table_column_names_original" in self.original_table:
            for tctc in self.original_table["table_column_names_original"]:
                self.original_table["tc_fast"].append(tctc[1].lower())
        else:
            for col in self.original_table['column_names_original']:
                if col[0] >= 0:
                    self.original_table["tc_fast"].append((self.original_table['table_names_original'][col[0]]+'.'+col[1]).lower())
                else:
                    self.original_table["tc_fast"].append("*")
        self.original_table["tc_fast_st"] = dict()
        for tctc in self.original_table["table_names_original"]:
            self.original_table["tc_fast_st"][tctc.lower()] = []
        for tctc in self.original_table["column_names"]:
            self.original_table["tc_fast_st"][self.original_table["table_names_original"][tctc[0]].lower()].append(tctc[1].lower())
        for i in range(len(self.original_table['table_names_original'])):
            self.original_table['table_names_original'][i] = self.original_table['table_names_original'][i].lower()

        self.column_tokens = [_tokenizer.tokenize(col[1]) for col in table_dict["column_names"]]
        self.table_tokens  = [_tokenizer.tokenize(ta)  for ta in table_dict["table_names"]]
        self.column_tokens_table_idx = [col[0] for col in table_dict["column_names"]]
        
        self.column_tokens_text_str = [col[1] for col in table_dict["column_names"]]
        self.column_tokens_lemma_str = [ " ".join([tok.lemma_ for tok in col]) for col in self.column_tokens ]
        self.column_tokens_lemma_str_tokens = [ col.split(" ") for col in self.column_tokens_lemma_str ]

        self.table_tokens_lemma_str = [ " ".join([tok.lemma_ for tok in col]) for col in self.table_tokens ]
        self.table_tokens_text_str = table_dict["table_names"]
        self.all_table_tokens_lemma_str = set([c for tok in self.table_tokens_lemma_str for c in tok.split(" ")])
        self.all_table_tokens_text_str = set([c for tok in self.table_tokens_text_str for c in tok.split(" ")])

        self.primaryKey = table_dict["primary_keys"]
        self.foreignKey = list(set([ j for i in table_dict['foreign_keys'] for j in i]))
        self.foreignKeyDict = dict()
        for fk in table_dict['foreign_keys']:
            if fk[0] not in self.foreignKeyDict.keys():
                self.foreignKeyDict[fk[0]] = [fk[1]]
            else:
                self.foreignKeyDict[fk[0]].append(fk[1])
            if fk[1] not in self.foreignKeyDict.keys():
                self.foreignKeyDict[fk[1]] = [fk[0]]
            else:
                self.foreignKeyDict[fk[1]].append(fk[0])

        self.db_id = table_dict["db_id"]
        self.column_types = table_dict['column_types']

        self.table_names_original  = table_dict["table_names_original"]
        self.column_names_original = table_dict["column_names_original"]

        self._concept_word = _concept_word
        self._tokenizer = _tokenizer
        if not stemmer:
            stemmer = MyStemmer()
        self._stemmer = stemmer
        if "same_col_idxs" in table_dict: 
            self.same_col_idxs = table_dict["same_col_idxs"]
        else:
            self.same_col_idxs = [[]]*len(table_dict["column_names"])

        self.table_col_text  = {-1:set()}
        self.table_col_lemma = {-1:set()}
        self.table_col_nltk  = {-1:set()}
        for i in range(len(table_dict["table_names"])):
            self.table_col_text[i] = set()
            self.table_col_lemma[i] = set()
            self.table_col_nltk[i] = set()
        for col,ocol in zip(self.column_tokens,table_dict["column_names"]):
            for tok in col:
                stem_tmp = stemmer.stem(tok.lower_)
                self.table_col_text[-1].add(tok.lower_)
                self.table_col_lemma[-1].add(tok.lemma_)
                self.table_col_nltk[-1].add(stem_tmp)
                self.table_col_text[ocol[0]].add(tok.lower_)
                self.table_col_lemma[ocol[0]].add(tok.lemma_)
                self.table_col_nltk[ocol[0]].add(stem_tmp)

        self.column_tokens_stem_str = [ " ".join([stemmer.stem(tok.text) for tok in col]) for col in self.column_tokens ]
        for i in range(len(self.column_tokens_stem_str)):
            for j in range(i+1,len(self.column_tokens_stem_str),1):
                if self.column_tokens_stem_str[i] == self.column_tokens_stem_str[j] and self.column_tokens_text_str[i] != self.column_tokens_text_str[j] and self.column_tokens_lemma_str[i] != self.column_tokens_lemma_str[j]:
                    stem_word = self.column_tokens_stem_str[i]
                    for z in range(i,len(self.column_tokens_stem_str),1):
                        if self.column_tokens_stem_str[z] == stem_word:
                            self.column_tokens_stem_str[z] = self.column_tokens_text_str[z]
        

        self.tbl_col_tokens_text_str = {}
        self.tbl_col_tokens_lemma_str = {}
        self.tbl_col_tokens_stem_str = {}
        self.tbl_col_idx_back = {}
        self.tbl_col_tokens_text_str_ori = {}

        self.tbl_col_tokens_text_str[-1] = self.column_tokens_text_str
        self.tbl_col_tokens_lemma_str[-1] = self.column_tokens_lemma_str
        self.tbl_col_tokens_stem_str[-1] = self.column_tokens_stem_str
        self.tbl_col_idx_back[-1] = [i for i in range(len(self.column_tokens_text_str))]
        self.tbl_col_tokens_text_str_ori[-1] = [i[1].lower() for i in self.column_names_original]


        for i in range(len(table_dict["table_names"])):
            self.tbl_col_tokens_text_str[i] = []
            self.tbl_col_tokens_lemma_str[i] = []
            self.tbl_col_tokens_stem_str[i] = []
            self.tbl_col_idx_back[i] = []
            self.tbl_col_tokens_text_str_ori[i] = []

        for tid,text,lemma,stem,cid,cor in zip(self.column_tokens_table_idx, self.column_tokens_text_str, self.column_tokens_lemma_str, self.column_tokens_stem_str, self.tbl_col_idx_back[-1],self.tbl_col_tokens_text_str_ori[-1]):
            if tid >= 0:
                self.tbl_col_tokens_text_str[tid].append(text)
                self.tbl_col_tokens_lemma_str[tid].append(lemma)
                self.tbl_col_tokens_stem_str[tid].append(stem)
                self.tbl_col_idx_back[tid].append(cid)
                self.tbl_col_tokens_text_str_ori[tid].append(cor)

    def is_bridge_table(self,table_1,table_2,bridge_table):
        if table_1 == table_2 or table_1 == bridge_table or table_2 == bridge_table:
            return False
        is_bridge_table = False
        for net in self.original_table['network']:
            if len(net[1]) == 3 and table_1 in net[1] and table_2 in net[1] and bridge_table in net[1]:
                is_bridge_table = True
            if len(net[1]) == 2 and table_1 in net[1] and table_2 in net[1]:
                return False
        return is_bridge_table
        

    def add_lower_data(self,table_dict):
        self.table_names_original_low  = [t.lower() for t in table_dict["table_names_original"]]
        if 'table_column_names_original' in table_dict:
            self.table_column_names_original_low  = [t[1].lower() for t in table_dict['table_column_names_original']]
        else:
            tmp = [""]+self.table_names_original_low
            self.table_column_names_original_low  = [ tmp[col[0]+1]+"."+col[1].lower() if col[0] >= 0 else "*" for col in self.column_names_original]


    def primary_keys(self,table_id):
        for key in self.primaryKey:
            if self.column_tokens_table_idx[key] == table_id:
                return key
        return -1

    def table_star_idx(self,table_id):
        for i,idx in enumerate(self.column_tokens_table_idx):
            if idx == table_id:
                return i
        return -1

    def lemmanize(self, word):
        return " ".join([i.lemma_ for i in self._tokenizer.tokenize(word)])
        

    def stem(self, word):
        return " ".join([self._stemmer.stem(w) for w in word.split(" ")])



    def get_related_word(self,words,mini_weight=0):
        result = set()
        if not self._concept_word:
            return result
        if isinstance(words,str):
            words = words.lower()
            for c in COUNTRYS:
                if words in c:
                    return c
            words = [words]

        for word in words:
            word = word.lower()
            if word in self._concept_word.keys():
                re_words = self._concept_word[word]
                for w in re_words:
                    if w[0] not in words and w[1]>mini_weight:
                        result.add(w[0])
        return list(result)

    def agg_plus_agg_clean(self,col_names):
        if col_names[0] in AGG_WORDS and len(col_names) > 1:
            final_one = col_names[-1].split(" ")
            if final_one[0] in AGG_WORDS and len(final_one) > 1:
                final_last = " ".join([final_one[i] for i in range(1,len(final_one))])
                for i in range(0,len(col_names)-1):
                    col_names[i] = col_names[i] + " " + final_last
        return col_names

    def column_contain_word(self, word, table_idx = -1):
        word_stem = self._stemmer.stem(word)
        word_token = self.lemmanize(word)
        if isinstance(table_idx, list):
            for t in table_idx:
                if word in self.table_col_text[t] or word in self.table_col_lemma[t] or word_stem in self.table_col_nltk[t] or word_token in self.table_col_text[t] or word_token in self.table_col_lemma[t]:
                    return True
        else:
            if word in self.table_col_text[table_idx] or word in self.table_col_lemma[table_idx] or word_stem in self.table_col_nltk[table_idx] or word_token in self.table_col_text[table_idx] or word_token in self.table_col_lemma[table_idx]:
                return True
        return False
    
    def column_concept_contain_word(self, word, table_idx = -1):
        table_idxs = table_idx
        if not isinstance(table_idx,list):
            table_idxs = [table_idx]
        for table_idx in table_idxs:
            words = self.get_related_word(word)
            for w in words:
                if w in self.table_col_text[table_idx] or w in self.table_col_lemma[table_idx] or self._stemmer.stem(w) in self.table_col_nltk[table_idx]:
                    return True
        return False

    def exact_match_potential_col(self, table_idx, word):
        re_list = set()
        word = self._tokenizer.tokenize(word)
        for i, col_lemma in enumerate(self.column_tokens_lemma_str):
            if table_idx < 0 or self.column_tokens_table_idx[i] == table_idx:
                for w in word:
                    if w.lemma_ in col_lemma:
                        re_list.add(i)
        return re_list


    def one_word_to_tables_column_match(self, table_idxs, word, table_in_this_col=False,cross_table_search=True,use_concept_match=True,final_round=True,only_two_match_fuc=False,allow_list=False):
        """
        table_in_this_col: means I know which table I am going to use.
        """
        agg = 0
        cols = []
        if -1 in table_idxs:
            table_idxs = [i for i in range(len(self.table_tokens_lemma_str))]
        if word.startswith("#"):
            word = word.replace("#","")
            match_function = [self.one_word_to_column_exact_match]
        elif final_round:
            match_function = [self.one_word_to_column_exact_match, self.one_word_to_column_exact_contain_match, self.one_word_to_column_agg_match,self.one_word_to_column_easy_contain_match, self.one_word_to_column_conceptnet_match]
        elif only_two_match_fuc:
            match_function = [self.one_word_to_column_exact_match, self.one_word_to_column_exact_contain_match]
        else:
            match_function = [self.one_word_to_column_exact_match, self.one_word_to_column_exact_contain_match, self.one_word_to_column_agg_match]

        # table.* : 
        for table_idx in table_idxs:
            if table_in_this_col and table_idx >= 0 and (word == "number" or word in INFORMATION_WORDS):
                m = self.table_star_idx(table_idx)
                if m >= 0:
                    cols.append(m)
        if cols:   
            if word == "number":
                return 3,cols
            return 0,cols
        

        for f in match_function:
            if not use_concept_match and f == self.one_word_to_column_conceptnet_match:
                continue
            for table_idx in table_idxs:
                a, c = f(table_idx, word, allow_list)
                if c != -2:
                    agg = a
                    if type(c) == list:
                        cols.extend(c)
                    else:
                        cols.append(c)
            if cols:
                return agg,cols

        if only_two_match_fuc:
            return agg,cols

        # remove or add table name to match
        for table_idx in table_idxs:
            new_word = None
            if self.in_outside_words(word,self.table_tokens_lemma_str[table_idx]):
                new_word = self.replace_to_delete_word(word,self.table_tokens_lemma_str[table_idx],"").strip()
                new_word = self.replace_to_delete_word(new_word,self.table_tokens_text_str[table_idx],"").strip()
            else:
                if " | " in self.table_tokens_lemma_str[table_idx]:
                    new_word = self.table_tokens_lemma_str[table_idx].split(" | ")[0] + " " + word
                else:
                    new_word = self.table_tokens_lemma_str[table_idx] + " " + word

            a, c = self.one_word_to_column_exact_match(table_idx, new_word)
            if c != -2:
                agg = a
                cols.append(c)
        if cols:
            return agg,cols
        
        if word in AGG_WORDS:
            return AGG_OPS[AGG_WORDS.index(word)], []
        
        input_lemma = self.lemmanize(word)
        if input_lemma != word:
            for i in table_idxs:
                if input_lemma in self.table_col_lemma[i]:
                    return self.one_word_to_several_column_exact_contain_match(i,input_lemma)

        if len(table_idxs) < len(self.table_tokens_lemma_str) and cross_table_search:
            rest_table = []
            for i in range(len(self.table_tokens_lemma_str)):
                if i not in table_idxs:
                    rest_table.append(i)
            
            for f in match_function:
                if not use_concept_match and f == self.one_word_to_column_conceptnet_match:
                    continue
                for table_idx in rest_table:
                    a, c = f(table_idx, word)
                    if c != -2:
                        agg = a
                        cols.append(c)
                if cols:
                    return agg,cols
        return agg, cols


    def one_word_to_column_exact_match(self, table_idx, word, allow_list=False):
        def exact_match(self, word,coll):
            if word.count(" ") == 1:
                ws = word.split(" ")
                if self.equal(word,coll) or self.equal(ws[1]+" "+ws[0],coll):
                    return True
            elif self.equal(word,coll):
                return True
            return False
        return_list = set()
        for j, coll in enumerate(self.tbl_col_tokens_text_str[table_idx]):
            if exact_match(self, word,coll):# word == coll:
                if table_idx < 0:
                    return_list.add(j)
                else:
                    return 0,self.tbl_col_idx_back[table_idx][j]
        
        word_lemma = self.lemmanize(word)
        for j, coll in enumerate(self.tbl_col_tokens_lemma_str[table_idx]):
            if exact_match(self, word_lemma,coll):
                if table_idx < 0:
                    return_list.add(j)
                else:
                    return 0,self.tbl_col_idx_back[table_idx][j]

        word_stem = self.stem(word)
        for j, coll in enumerate(self.tbl_col_tokens_stem_str[table_idx]):
            if exact_match(self, word_stem,coll):
                if table_idx < 0:
                    return_list.add(j)
                else:
                    return 0,self.tbl_col_idx_back[table_idx][j]

        if len(return_list) == 1:
            return 0,self.tbl_col_idx_back[table_idx][list(return_list)[0]]
        elif return_list and allow_list:
            return 0,[self.tbl_col_idx_back[table_idx][i] for i in return_list]
        elif return_list:
            return -2,-2
        return 0,-2


    def one_word_to_column_exact_contain_match(self, table_idx, word, allow_list=False):
        word = self.lemmanize(word)
        re_ = []
        # word is part of the column string but do not split the word
        for i, col_lemma in enumerate(self.column_tokens_lemma_str_tokens):
            if table_idx < 0 or self.column_tokens_table_idx[i] == table_idx:
                if word in col_lemma and "*" not in col_lemma:
                    re_.append(i)

        if len(re_) == 1:
            return 0,re_[0]
        elif re_ and allow_list:
            return 0,re_

        return self.one_word_to_column_easy_contain_match(table_idx, word, easy=False)


    def one_word_to_column_easy_contain_match(self, table_idx, word, easy=True, allow_list=False):
        word = self._tokenizer.tokenize(word)
        if len(word) > 1:
            # word is part of the column string but split the word
            # that is why here need len(word) > 1.
            re_count = [0 for i in range(len(self.column_tokens_lemma_str))]
            for i, col_lemma in enumerate(self.column_tokens_lemma_str_tokens):
                if table_idx < 0 or self.column_tokens_table_idx[i] == table_idx:
                    for w in word:
                        if w.lemma_ in col_lemma and "*" not in col_lemma:
                            re_count[i] += 1
            if easy: # only part of word in column string and this part is most in all col
                if re_count.count(max(re_count)) == 1:
                    return 0,re_count.index(max(re_count))  
            else:   # all splited word in column string
                if max(re_count) == len(word):
                    if re_count.count(max(re_count)) == 1:
                        return 0,re_count.index(max(re_count))
                    for i,o in enumerate(re_count):
                        if o == len(word): 
                            if o == len(self.column_tokens_text_str[i].split(" | ")[0].split(" ")):
                                return 0,i

        return 0,-2

    
    def one_word_to_column_conceptnet_match(self, table_idx, word, allow_list=False):
        return_list = set()
        if word == "number":
            return 0,-2
        if self.stem(word) in self.table_col_nltk[-1]:
            return 0,-2
        # concept net:
        words = self.get_related_word(word)
        for j, coll in enumerate(self.column_tokens_lemma_str):
            if table_idx < 0 or self.column_tokens_table_idx[j] == table_idx:
                for w in words:
                    if self.equal(w,coll):
                        if table_idx < 0:
                            return_list.add(j)
                        else:
                            return 0,j
        if len(return_list) == 1:
            return 0,list(return_list)[0]
        return 0,-2


    def one_word_to_column_agg_match(self, table_idx, word, allow_list=False):
        def agg_in_words(words):
            for i, w in enumerate(words):
                if w in AGG_WORDS:
                    if i + 1 < len(words):
                        return (" ".join(words[0:i])+" "+" ".join(words[i+1:])).strip(),AGG_OPS[AGG_WORDS.index(words[i])]
                    return (" ".join(words[0:i])).strip(),AGG_OPS[AGG_WORDS.index(words[i])]
            return None,0
        if word == "number" and table_idx >= 0:
            return 3, self.column_tokens_table_idx.index(table_idx)
        word = word.split(" ")

        if len(word) > 1:
            new_word,agg = agg_in_words(word)
            if new_word:
                match_function = [self.one_word_to_column_exact_match, self.one_word_to_column_exact_contain_match, self.one_word_to_column_easy_contain_match, self.one_word_to_column_conceptnet_match]
                for f in match_function:
                    _, m = f(table_idx, new_word)
                    if m != -2:
                        if agg == 4 and self.column_types[m] != "number":
                            agg = 0
                        return agg, m
        return 0, -2


    def one_word_to_several_column_exact_contain_match(self, table_idx, word):
        re_ = [-2]
        # word is part of the column string but do not split the word
        for i, col_lemma in enumerate(self.column_tokens_lemma_str_tokens):
            if table_idx < 0 or self.column_tokens_table_idx[i] == table_idx:
                if word in col_lemma and "*" not in col_lemma:
                    re_.append(i)
        return 0,re_

    def table_match(self, table_str):
        table_str = " ".join(table_str)
        for i, table in enumerate(self.table_tokens_text_str):
            if self.equal(table_str, table):
                return i
        
        toks = self._tokenizer.tokenize(table_str)
        table_str = " ".join([i.lemma_ for i in toks])

        for i, table in enumerate(self.table_tokens_lemma_str):
            if self.equal(table_str, table):
                return i
        re_ = -1
        re_count = 0
        for i, table in enumerate(self.table_tokens_lemma_str):
            if self.equal(table_str, table):
                re_ = i
                re_count += 1
        if re_count == 1:
            return re_
        
        if len(toks) > 1: # easy contain match
            re_count = [0 for i in range(len(self.table_tokens_lemma_str))]
            for i, table_lemma in enumerate(self.table_tokens_lemma_str):
                for w in toks:
                    if w.lemma_ in table_lemma:
                        re_count[i] += 1
            if re_count.count(max(re_count)) == 1:
                    return re_count.index(max(re_count))
        return -1

    def equal(self, str_outside, str_inside):
        if " | " in str_inside:
            strs = str_inside.split(" | ")
            for s in strs:
                if s == str_outside:
                    return True
        elif str_outside == str_inside:
            return True
        return False

    def in_outside_words(self, str_outside, str_inside):
        if " | " in str_inside:
            strs = str_inside.split(" | ")
            for s in strs:
                if s in str_outside:
                    return True
        elif str_inside in str_outside:
            return True
        return False
    
    def replace_to_delete_word(self, str_outside,str_inside,new_word=""):
        if " | " in str_inside:
            strs = str_inside.split(" | ")
            for s in strs:
                str_outside = str_outside.replace(s,"")
        else:
            str_outside = str_outside.replace(str_inside,"")
        return str_outside