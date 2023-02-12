import nltk
from .match import ALL_JJS

DICT = {"weight":"weigh",
"won":"win",
"nation":"country",

}

class MyStemmer():
    def __init__(self):
        self.stemmer = nltk.stem.LancasterStemmer()
    
    def stem(self,w):
        result = w.lower()
        if result == "january":
            return "jan"
        elif result == "february":
            result = "feb"
        elif result == "march":
            return "mar"
        elif result == "april":
            return "apr"
        elif result == "may":
            return "may"
        elif result == "june":
            return "jun"
        elif result == "july":
            return "jul"
        elif result == "august":
            return "aug"
        elif result == "september":
            return "sep"
        elif result == "sept":
            return "sep"
        elif result == "october":
            return "oct"
        elif result == "november":
            return "nov"
        elif result == "december":
            return "dec"
        result = self.stemmer.stem(result)
        if result == "weight":
            result = "weigh"
        if result == "hight":
            result = "high"
        elif result == "won":
            result = "win"
        elif result in ALL_JJS:
            return ALL_JJS[result]
        elif result == "maxim":
            result = "max"
        elif result == "minim":
            result = "min"
        return result