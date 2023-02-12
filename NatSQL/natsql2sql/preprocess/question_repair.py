import json
import pickle
from spacy.symbols import ORTH, LEMMA
import editdistance
from .TokenString import SToken as Token
from .match import word_is_grsm,ABSOLUTELY_GRSM_DICT,ABSOLUTELY_GREATER,ABSOLUTELY_SMALLER,NOT_STAR_WORD
from .utils import str_is_num,is_there_sgrsm_and_gr_or_sm


def cover_by_punt(tokens,i):
    punt = False
    for j in range(i,i+5,1):
        if j >= len(tokens):
            break
        if tokens[j].text == "'":
            punt = True
            break
    if punt:
        for j in range(i,i-5,-1):
            if j < 0:
                break
            if tokens[j].text == "'":
                punt = False
                break
        if not punt:
            return True
    return False

def switch_tok_match(word,target):
    len_w = len(word)
    for i in range(len_w-1):
        nw = word[0:i] + word[i+1] + word[i] + word[i+2:]
        if nw in target:
            return nw
    return None



def question_repair(nlp,q,all_word,full_target,top_target,end_target):
    full_target.extend(list(NOT_STAR_WORD))
    q = q.replace('‘',"'")
    q = q.replace('’',"'")
    q = q.replace('('," ( ")
    q = q.replace(')'," ) ")
    q = q.replace('"',"'")
    q = q.replace('  '," ")
    q = q.replace('  '," ")
    q = q.replace('  '," ")

    q = q.replace(" (at least one) "," at least one ")
    q = q.replace(" ( at least one ) "," at least one ")
    q = q.replace(" at least once "," at8least8one ")
    q = q.replace(" at least 1 "," at8least8one ")
    q = q.replace(" at least a "," at8least8one ")
    q = q.replace(" at least an "," at8least8one ")
    q = q.replace(" one or more "," at8least8one ")
    q = q.replace(" some at least one "," at8least8one ")
    q = q.replace(" than at least one "," than a ") 
    q = q.replace(" in at least one "," in a ") 
    q = q.replace(" at least one "," at8least8one ")
    
    q = q.replace(" (at least one)."," at least one.")
    q = q.replace(" ( at least one )."," at least one.")
    q = q.replace(" at least once."," at8least8one.")
    q = q.replace(" at least 1."," at8least8one.")
    q = q.replace(" at least a."," at8least8one.")
    q = q.replace(" at least an."," at8least8one.")
    q = q.replace(" one or more."," at8least8one.")
    q = q.replace(" some at least one."," at8least8one.")
    q = q.replace(" than at least one."," than a.") 
    q = q.replace(" in at least one."," in a.") 
    q = q.replace(" at least one."," at8least8one.")
    
    q = q.replace(" (at least one)?"," at least one?")
    q = q.replace(" ( at least one )?"," at least one?")
    q = q.replace(" at least once?"," at8least8one?")
    q = q.replace(" at least 1?"," at8least8one?")
    q = q.replace(" at least a?"," at8least8one?")
    q = q.replace(" at least an?"," at8least8one?")
    q = q.replace(" one or more?"," at8least8one?")
    q = q.replace(" some at least one?"," at8least8one?")
    q = q.replace(" than at least one?"," than a?") 
    q = q.replace(" in at least one?"," in a?") 
    q = q.replace(" at least one?"," at8least8one?")

    q = q.replace(" (at least one),"," at least one,")
    q = q.replace(" ( at least one ),"," at least one,")
    q = q.replace(" at least once,"," at8least8one,")
    q = q.replace(" at least 1,"," at8least8one,")
    q = q.replace(" at least a,"," at8least8one,")
    q = q.replace(" at least an,"," at8least8one,")
    q = q.replace(" one or more,"," at8least8one,")
    q = q.replace(" some at least one,"," at8least8one,")
    q = q.replace(" than at least one,"," than a,") 
    q = q.replace(" in at least one,"," in a,") 
    q = q.replace(" at least one,"," at8least8one,")
  

    sentences = nlp(q)
    tokens = [t for t in sentences]
    sentences_num = len(list(sentences.sents))
    if sentences_num == 2 and tokens[0].lemma_ == "which" and tokens[2].lemma_ == "have":
        tokens[2] = Token(text="has",lemma="have")
    for i,tok in enumerate(tokens) :
        if tok.lemma_ not in all_word and tok.lower_ not in all_word and not tok.text.isdigit() and tok.text.islower():
            if tok.lemma_ not in full_target and tok.lower_ not in full_target:
                switch_fix_text = switch_tok_match(tok.text,full_target)
                switch_fix_lemma = switch_tok_match(tok.lemma_,full_target)
                for w in full_target:
                    if len(w)>3 and (editdistance.eval(tok.text,w) == 1 or w==switch_fix_text):
                        if cover_by_punt(tokens,i):
                            continue
                        tokens[i] = Token(text=w,lemma=w,tag =tok.tag_)
                        print(tok.text + ' -> ' + tokens[i].text)
                        break
                    elif len(w)>3 and (editdistance.eval(tok.lemma_,w) == 1 or w==switch_fix_lemma):
                        if cover_by_punt(tokens,i):
                            continue
                        tokens[i] = Token(text=w,lemma=w,tag =tok.tag_)
                        print(tok.text + ' -> ' + tokens[i].text)
                        break
                    elif tok.text == w+"id" or tok.text == w+"ids" or tok.lemma_ == w+"id" or tok.lemma_ == w+"ids":
                        if cover_by_punt(tokens,i):
                            continue
                        tokens[i] = Token(text=w+" id",lemma=w + " id",tag =tok.tag_)
                        print(tok.text + ' -> ' + tokens[i].text)
                        break
                    elif len(w)>8 and editdistance.eval(tok.lemma_,w) == 2 and len(w) - len(tok.lemma_)  == 2:
                        if cover_by_punt(tokens,i):
                            continue
                        for tok_i in range(2,len(tok.lemma_)-1,1):
                            if editdistance.eval(tok.lemma_[:tok_i] + tok.lemma_[tok_i] + tok.lemma_[tok_i:], w) == 1:
                                tokens[i] = Token(text=w,lemma=w,tag =tok.tag_)
                                print(tok.text + ' -> ' + tokens[i].text)
                                break
                        break

        if i + 2 < len(tokens) and tok.tag_ in ["JJR","RBR"] and tokens[i+1].text.isdigit():
            tokens.insert(i+1,Token(text="than"))

        elif tok.lemma_ == "or" and i + 2 < len(tokens) and i > 3:
            
            if tokens[i-1].lemma_ == "than" and word_is_grsm(tokens[i-2]) and tokens[i+1].lemma_ == "equal":
                if tokens[i+2].lemma_ == "to":
                    del tokens[i+2]
                find_sgrsm = is_there_sgrsm_and_gr_or_sm(tokens,tokens[i-2],i-2)
                if find_sgrsm == "GR_" or (find_sgrsm is None and tokens[i-2].lemma_ in ABSOLUTELY_GREATER):
                    tokens[i-2] = Token(text="shinier")
                else:
                    tokens[i-2] = Token(text="uglier")
                del tokens[i]
                del tokens[i]
                del tokens[i-1]
            elif str_is_num(tokens[i-1].text) and (word_is_grsm(tokens[i+1]) or tokens[i+1].text in ["young","younger","old","older","before","late","later","after"]) and i >= 4 and not word_is_grsm(tokens[i-2]) and not word_is_grsm(tokens[i-3]) and not word_is_grsm(tokens[i-4]):
                tokens[i] = tokens[i-1]
                find_sgrsm = is_there_sgrsm_and_gr_or_sm(tokens,tokens[i+1],i+1)
                if find_sgrsm == "GR_" or (find_sgrsm is None and tokens[i+1].lemma_ in ABSOLUTELY_GREATER):
                    tokens[i-1] = Token(text="shinier")
                else:
                    tokens[i-1] = Token(text="uglier")
                if tokens[i-2].lemma_ == "aged":
                    tokens[i+1] = tokens[i]
                    tokens[i] = tokens[i-1]
                    tokens[i-1] = Token(text="age")
                    tokens[i-2] = Token(text="whose")
                else:
                    del tokens[i+1]
            elif tokens[i-1].text in ["in"] and tokens[i+1].lemma_ in ABSOLUTELY_GRSM_DICT.keys() and str_is_num(tokens[i+2].text):
                find_sgrsm = is_there_sgrsm_and_gr_or_sm(tokens,tokens[i+1],i+1)
                if find_sgrsm == "GR_" or (find_sgrsm is None and tokens[i+1].lemma_ in ABSOLUTELY_GREATER):
                    tokens[i-1] = Token(text="shinier")
                else:
                    tokens[i-1] = Token(text="uglier")
                del tokens[i]
                del tokens[i]
           
    return tokens




