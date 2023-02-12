from functools import lru_cache
AGG_WORDS = ["average","maximum","minimum","number"]

class SToken():
    def __init__(self,
                 text: str = None,
                 idx: int = None,
                 lemma: str = None,
                 pos: str = None,
                 tag: str = "NN",
                 dep: str = None,
                 ent_type: str = None,
                 text_id: int = None,
                 lower = None,
                 head = None) -> None:
        self.text = text
        self.idx = idx
        self.lemma_ = lemma
        self.pos_ = pos
        self.tag_ = tag
        self.dep_ = dep
        self.ent_type_ = ent_type
        self.text_id = text_id
        self.lower_ = lower
        self.head = head
        if text and not lower:
            self.lower_ = text.lower()
        self.lemma_ = lemma
        if text and not lemma:
            self.lemma_ = self.lower_

    def replace(self,old,new):
        self.text = self.text.replace(old,new)
        self.lemma_ = self.lemma_.replace(old,new)
        self.lower_ = self.lower_.replace(old,new)
        return self

    def __repr__(self):
         return self.text



class TokenString():
    def __init__(self, tokenizer, str_, for_copy = None):
        if str_:
            if tokenizer:
                self.tokens = tokenizer.tokenize(str_)
            else:
                self.tokens = str_
            self.text = " ".join([tok.text for tok in self.tokens]).lower()
            self.lemma_ = " ".join([tok.lemma_ if tok.lemma_ != '-PRON-' else tok.text for tok in self.tokens]).lower()
            self.tag_ = " ".join([tok.tag_ for tok in self.tokens])
        if for_copy:
            self.tokens = for_copy[0]
            self.text   = for_copy[1]
            self.lemma_ = for_copy[2]
            self.tag_   = for_copy[3]

    def __len__(self):
        return len(self.tokens)
    
    def __str__(self):
        return self.text
    
    def __deepcopy__(self,d):
        return TokenString(None,None,(self.tokens,self.text,self.lemma_,self.tag_))

    
    def match_all(self, off, in_toks, total_tokens, type_=1):
        """
        type_:  if it is 1, 
                    it will match based on str.lower(); 
                if it is 2, 
                    it will match based on str.lemma_; 
                else it will match based on the tag_.
        """
        if type_ == 1:
            for i in range(len(in_toks)):
                if in_toks[i] != total_tokens[i+off].lower_:
                    return False
        elif type_ == 2:
            for i in range(len(in_toks)):
                lemma = total_tokens[i+off].lemma_ if total_tokens[i+off].lemma_ != '-PRON-' else total_tokens[i+off].text
                if in_toks[i] != lemma:
                    return False
        else:
            for i in range(len(in_toks)):
                if in_toks[i] != total_tokens[i+off].tag_:
                    return False
        return True


    def re_construct(self, str_, type_ = 1):
        """
        type_:  if it is 1, 
                    it will re construct based on str.lower(); 
                if it is 2, 
                    it will re construct based on str.lemma_; 
                else it will re construct based on the tag_.
        """
        str_tokens = str_.strip().split(" ")
        len_str_tokens = len(str_tokens)
        target_tokens = None

        for i,tok in enumerate(self.tokens):
            if i + len_str_tokens > len(self.tokens):
                break
            if self.match_all(i, str_tokens, self.tokens, type_):
                return self.construct_from_token(self.tokens[i:len_str_tokens+i])
               
    def index(self, str_, strat_index=0, type_=1):
        """
        type_:  if it is 1, 
                    it will return index based on str.lower(); 
                if it is 2, 
                    it will return index based on str.lemma_; 
                else it will return index based on the tag_.
        """
        str_tokens = str_.strip().split(" ")
        len_str_tokens = len(str_tokens)
        target_tokens = None
        for i,tok in enumerate(self.tokens):
            if i < strat_index:
                continue
            if i + len_str_tokens > len(self.tokens):
                break
            if self.match_all(i, str_tokens, self.tokens, type_):
                return i
        return -1
    
    def add_token_string(self,token_string):
        if token_string:
            self.tokens.extend(token_string.tokens)
            self.text += " " + token_string.text
            self.lemma_ += " " + token_string.lemma_
            self.tag_ += " " + token_string.tag_


    def update(self, idx, tok):
        self.tokens[idx] = tok
        self.text = " ".join([tok.text for tok in self.tokens]).lower()
        self.lemma_ = " ".join([tok.lemma_ if tok.lemma_ != '-PRON-' else tok.text  for tok in self.tokens]).lower()
        self.tag_ = " ".join([tok.tag_ for tok in self.tokens])


    def split(self, split_word, allowed_split_to_many = False):
        if split_word in self.text and (allowed_split_to_many or self.text.count(split_word) == 1):
            str_ = self.text.split(split_word)
            if not allowed_split_to_many:
                str_left  = self.re_construct(str_[0])
                str_right = self.re_construct(str_[1])
                return (str_left, str_right)
            else:
                final_list = []
                for str_one in str_:
                    final_list.append(self.re_construct(str_one))
                return final_list
        return None

    
    def clean_stop_word(self):
        for i in range(len(self.tokens)-1,0,-1):
            if self.tokens[i].lemma_ in STOP_WORDS or self.tokens[i].text in STOP_WORDS:
                del self.tokens[i]
        self.refresh()

    def count(self, word):
        return self.text.count(word)

    def clean_punctuation(self):
        for i in range(len(self.tokens)-1,0,-1):
            if not self.tokens[i].text.isalpha():
                del self.tokens[i]
        self.refresh()

    def replace(self,oldword,new_word):
        newself = copy.deepcopy(self)
        for i, tok in enumerate(newself.tokens):
            if tok.lemma_ == oldword or tok.text == oldword:
                newself.tokens[i] = SToken(text=new_word,lemma=new_word,tag=tok.tag_,lower=new_word.lower())
        if new_word == "":
            newself.remove_token("")
        newself.refresh()
        return newself

    def remove_token(self, word):
        for i in range(len(self.tokens)-1,0,-1):
            if self.tokens[i].lemma_ == word or self.tokens[i].text == word:
                del self.tokens[i]
        self.refresh()
    
    def refresh(self):
        self.text = " ".join([tok.text for tok in self.tokens]).lower()
        self.lemma_ = " ".join([tok.lemma_ if tok.lemma_ != '-PRON-' else tok.text for tok in self.tokens]).lower()
        self.tag_ = " ".join([tok.tag_ for tok in self.tokens])
    

    def delete_suffix(self):
        for i,tok in enumerate(self.tokens):
            if tok.lemma_.endswith("ing") and tok.text == tok.lemma_:
                self.tokens[i] = SToken(tok.text,tok.idx,tok.lemma_[:-3],tok.pos_,tok.tag_,tok.dep_,tok.ent_type_,0,tok.lower_,tok.head)
                self.text = " ".join([tok.text for tok in self.tokens]).lower()
                self.lemma_ = " ".join([tok.lemma_ if tok.lemma_ != '-PRON-' else tok.text for tok in self.tokens]).lower()
                self.tag_ = " ".join([tok.tag_ for tok in self.tokens])
            elif tok.lemma_.endswith("es") and tok.text == tok.lemma_:
                self.tokens[i] = SToken(tok.text,tok.idx,tok.lemma_[:-2],tok.pos_,tok.tag_,tok.dep_,tok.ent_type_,0,tok.lower_,tok.head)
                self.text = " ".join([tok.text for tok in self.tokens]).lower()
                self.lemma_ = " ".join([tok.lemma_ if tok.lemma_ != '-PRON-' else tok.text for tok in self.tokens]).lower()
                self.tag_ = " ".join([tok.tag_ for tok in self.tokens])

    def lemma_without_jjs_jjr(self):
        return " ".join([tok.lemma_ if tok.lemma_ != '-PRON-' and tok.tag_ not in ["JJR","JJS","RBR","RBS"] else tok.text for tok in self.tokens]).lower()


    @classmethod
    def construct_from_token(self,tokens):
        ts = TokenString(None,None)
        ts.tokens = tokens
        ts.text = " ".join([tok.text for tok in ts.tokens]).lower()
        ts.lemma_ = " ".join([tok.lemma_ if tok.lemma_ != '-PRON-' else tok.text for tok in ts.tokens]).lower()
        ts.tag_ = " ".join([tok.tag_ for tok in ts.tokens])
        return ts



class Tokenizer_Similar_Allennlp():
    def __init__(self, spacy):
        self.spacy = spacy

    def tokenize(self, str_):
        return [tok for tok in self.spacy(str_)]



global_tokenizer = None
global_spacy = None



def get_spacy_tokenizer():
    global global_tokenizer
    global global_spacy

    if global_tokenizer:
        return global_tokenizer

    import spacy
    from spacy.symbols import ORTH, LEMMA
    nlp = spacy.load("en_core_web_sm")
    import re
    from spacy.tokenizer import Tokenizer

    suffixes = nlp.Defaults.suffixes +  (r'((\d{4}((_|-|/){1}\d{2}){2})|((\d{2})(_|-|/)){2}\d{4})(\s\d{2}(:\d{2}){2}){0,1}',) + (r'(\d{1,2}(st|nd|rd|th){0,1}(,|\s)){0,1}((J|j)an(uary){0,1}|(F|f)eb(ruary){0,1}|(M|m)ar(ch){0,1}|(A|a)pr(il){0,1}|(M|m)ay|(J|j)un(e){0,1}|(J|j)ul(y){0,1}|(A|a)ug(ust){0,1}|(S|s)ep(tember){0,1}|(O|o)ct(ober){0,1}|(N|n)ov(ember){0,1}|(D|d)ec(ember){0,1})(\s|,)(\d{1,2}(st|nd|rd|th){0,1}(\s|,){1,3}){0,1}\d{4}',) + ( r'(\d{1,6}(_|-|\+|/)\d{0,6}[A-Za-z]{0,6}\d{0,6}[A-Za-z]{0,6})',)
    suffix_regex = spacy.util.compile_suffix_regex(suffixes)
    nlp.tokenizer.suffix_search = suffix_regex.search
    
    nlp.tokenizer.add_special_case(u'Ph.D', [{ORTH: u'Ph.D', LEMMA: u'ph.d'}])
    nlp.tokenizer.add_special_case(u'id', [{ORTH: u'id', LEMMA: u'id'}])
    nlp.tokenizer.add_special_case(u'Id', [{ORTH: u'Id', LEMMA: u'id'}])
    nlp.tokenizer.add_special_case(u'ID', [{ORTH: u'ID', LEMMA: u'id'}])
    nlp.tokenizer.add_special_case(u'iD', [{ORTH: u'iD', LEMMA: u'id'}])
    nlp.tokenizer.add_special_case(u'statuses', [{ORTH: u'statuses', LEMMA: u'status'}])

    global_tokenizer = Tokenizer_Similar_Allennlp(nlp)
    global_spacy = nlp
    return global_tokenizer

@lru_cache(maxsize=5000)
def lemmatization(s):
    tokenizer = get_spacy_tokenizer()
    return ("").join([tok.lemma_ for tok in tokenizer.tokenize(s)])
