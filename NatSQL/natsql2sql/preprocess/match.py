import copy

# The STOP_WORDS and WHERE_STOP_WORDS are adapted from https://github.com/benbogin/spider-schema-gnn/blob/02f4ae43b891f41909215e889e37fbc084f982e1/semparse/contexts/spider_db_context.py

STOP_WORDS = {"", "'", "all", "being", "-", "over", "through", "yourselves", "its", "before",
              "hadn", "with", "had", ",", "should", "to", "only", "under", "ours", "has", "ought", "do",
              "them", "his", "than", "very", "cannot", "they", "not", "during", "yourself", "him",
              "nor", "did", "didn", "'ve", "this", "she", "where", "because", "doing", "some", "we", "are",
              "further", "ourselves", "out", "what", "for", "weren", "does", "above", "between", "mustn", "?",
              "be", "hasn", "who", "were", "here", "shouldn", "let", "hers", "by", "both", "about", "couldn",
              "of", "could", "against", "isn", "or", "own", "into", "while", "whom", "down", "wasn", "your",
              "from", "her", "their", "aren", "there", "been", ".", "few", "too", "wouldn", "themselves",
              ":", "was", "until", "more", "himself", "on", "but", "don", "herself", "haven", "those", "he",
              "me", "myself", "these", "up", ";", "below", "'re", "can", "theirs", "my", "and", "would", "then",
              "is", "am", "it", "doesn", "an", "as", "itself", "at", "have", "in", "any", "if", "!",
              "again", "'ll", "no", "that", "when", "same", "how", "other", "which", "you", "many", "shan",
              "'t", "n't","'s", "our", "after",  "'d", "such", "'m", "why", "a", "off", "i", "yours", "so",
              "the", "having", "once"
              ,"either","every","yet"}

NEGATIVE_WORDS = {"hasn",'except',"shouldn","mustn","isn","wasn", "aren","wouldn", "don","haven", "doesn","'t","hadn","not", "no", "cannot","nor", "didn", "weren", "couldn","without","never","n't","outside","null","empty"}
INFORMATION_WORDS = ["info","information","detail","infos","informations","details"]

WHERE_STOP_WORDS = {"",")","(", "'", "all", "being", "-", "over", "through", "yourselves", "its", 
            #   "hadn", 
              "with", "had", ",", "should", "to", "only", "ours", "has", "ought", "do",
              "them", "his", "very",  "they", "during", "yourself", "him",
               "did", "'ve", "this", "she",  "where", "because", "doing", "some", "we", "are",
              "ourselves", "out", "what", "for",  "does", "?",
              "be",  "who", "were", "here",  "let", "hers", "by", "both", "about", 
              "of", "could", "against",  "own", "while", "whom",  "your",
               "her", "their", "there", "been", ".", "too", "themselves",
              ":", "was", "until", "himself",  "but",  "herself", "those", "he",
              "me", "myself", "these", ";",  "'re", "can", "theirs", "my", "and", "or", "would", "then",
              "is", "am", "it",  "an",  "itself", "have", "in", "any", "if", "!",
              "again", "'ll",  "that", "when", "same", "how", "other", "which", "you", "many", "shan",
               "'s", "our", "'d", "such", "'m", "why", "a", "off", "i", "yours", "so",
              "the", "having", "once"
              ,"either","every","yet"}

SELECT_FIRST_WORD = {'list', 'how', 'which', 'find', 'return', 'what', 'show',
 'count', 'compute',  'give', 'tell' }#'Describe','Sort'

AGG_WORDS = ["average","maximum","minimum","number","total","mean","avg","max","min"]
AGG_OPS = [5,1,2,3,4,5,5,1,2]

ALL_IMPORTANT_PATTERN_TOKENS = {'last','start','first',"lexicographic","top",'SM_JJS','SM_SJJS','SM_GRSM','SM_SGRSM','GR_JJS','GR_SJJS','GR_GRSM','GR_SGRSM','word', 'and', 'one', 'only', 'less', 'IN', 'GPE', 'substring', 'alphabetical', 'also', 'SDB', 'UDB', 'V', 'phrase', 'NUM',  'each', 'NN', 'contain', 'except', 'SGRSM', 'COL',  'more', 'DB', 'AGG', 'most', 'WP', 'DATE', 'YEAR', 'RP', 'like', 'GRSM', 'order', 'english', 'to', 'amount', 'than', 'JJS', 'but', 'total', 'TO', 'top', 'CC', 'or', 'include', 'any', 'least', 'other', 'ascend', 'between', 'have', 'from', 'frequent', 'common', 'popular' , 'at', 'ascending', 'in', 'letter', 'by', 'TABLE', 'both', 'PDB', 'number', 'as', 'SM', 'descend', 'descending' 'of', 'JJ', 'string', 'SJJS', 'NOT', 'sort', 'GR',"TABLE-COL", "BCOL","multiple","shinier","uglier",">=","<=",">","<","tilt"}

NOT_STAR_WORD = {'last','first', 'one', 'only', 'substring', 'alphabetical', 'phrase', 'except', 'like', 'order', 'than', 'top',  'include', 'ascend', 'between', 'ascending', 'letter', 'descend', 'descending', 'each', 'sort', "group",\
    "lexicographic","lexicographical",'more','most','but', 'least','other','frequent', 'number',  }

ALL_SPECIAL_WORD = STOP_WORDS.union(NEGATIVE_WORDS).union(SELECT_FIRST_WORD).union(AGG_WORDS).union(ALL_IMPORTANT_PATTERN_TOKENS).union(NOT_STAR_WORD)

SYNONYM = {"day":["date"]
}

S_ADJ_WORD_DIRECTION = {
    # we can infer the column from these special adj
    # [col, agg, 0:desc|1:asc, 4:<|3:>]
    # 1 always go with 4
    "young":[["age",0,1,4],["birth",0,0,3],["birthday",0,0,3],["year",0,0,3],["found",0,0,3],["date",0,0,3]],
    "old":[["age",0,0,3],["birth",0,1,4],["birthday",0,1,4],["year",0,1,4],["found",0,1,4],["date",0,1,4]],

    "heavy":[["weight",0,0,3],["weigh",0,0,3],["kilogram",0,0,3],["kg",0,0,3],["gram",0,0,3],["g",0,0,3]],
    "fat":[["weight",0,0,3],["weigh",0,0,3],["kilogram",0,0,3],["kg",0,0,3],["gram",0,0,3],["g",0,0,3]],
    "light":[["weight",0,1,4],["weigh",0,1,4],["kilogram",0,1,4],["kg",0,1,4],["gram",0,1,4],["g",0,1,4]],
    

    "late":[["date",0,0,3],["year",0,0,3],["found",0,0,3]],
    "early":[["date",0,1,4],["year",0,1,4],["found",0,1,4]],

    "dominant":[["percentage",0,0,3]],
    "predominant":[["percentage",0,0,3]],
    
    "predominantly":[["percentage",0,0,3]],
    "popular":[["percentage",0,0,3]],
    "main":[["percentage",0,0,3]],
    "major":[["percentage",0,0,3]],
    "prime":[["percentage",0,0,3]],
    "primary":[["percentage",0,0,3]],
    "key":[["percentage",0,0,3]],

    "elderly":[["age",0,0,3],["birth",0,1,4],["birthday",0,1,4]],


    "long":[["length",0,0,3],["milliseconds",0,0,3],["second",0,0,3],["minute",0,0,3],["hour",0,0,3],["duration",0,0,3],["kilometer",0,0,3],["KM",0,0,3],["meter",0,0,3],["M",0,0,3],["distance",0,0,3],["inch",0,0,3],["centimeter",0,0,3],["cm",0,0,3],["date",0,1,4]],
    "short":[["length",0,1,4],["height",0,1,4],["milliseconds",0,1,4],["second",0,1,4],["minute",0,1,4],["hour",0,1,4],["distance",0,1,4],["duration",0,1,4],["kilometer",0,1,4],["KM",0,1,4],["meter",0,1,4],["M",0,1,4],["foot",0,1,4],["inch",0,1,4],["centimeter",0,1,4],["cm",0,1,4]],

    "longsong":[["duration",0,0,3]],
    "shortsong":[["duration",0,1,4]],


    "new":[["year",0,0,3],["moth",0,0,3],["date",0,0,3]],
    "previous":[["minute",0,1,4],["date",0,1,4],["year",0,1,4],["hour",0,1,4],["second",0,1,4],["day",0,1,4]],
    "ancient":[["date",0,1,4],["year",0,1,4],["day",0,1,4],["moth",0,1,4]],
    "prior":[["minute",0,1,4],["date",0,1,4],["year",0,1,4],["hour",0,1,4],["second",0,1,4],["day",0,1,4]],
    
    
    "quick":[["milliseconds",0,1,4],["second",0,1,4],["minute",0,1,4],["hour",0,1,4],["day",0,1,4],["year",0,1,4],["time",0,1,4],["speed",0,0,3]],
    "fast":[["milliseconds",0,1,4],["second",0,1,4],["minute",0,1,4],["hour",0,1,4],["day",0,1,4],["year",0,1,4],["time",0,1,4],["speed",0,0,3]],
    "slow":[["milliseconds",0,0,3],["second",0,0,3],["minute",0,0,3],["hour",0,0,3],["day",0,0,3],["year",0,0,3],["time",0,0,3],["speed",0,1,4]],


    "high":[["height",0,0,3],["kilometer",0,0,3],["KM",0,0,3],["meter",0,0,3],["M",0,0,3],["foot",0,0,3],["inch",0,0,3],["centimeter",0,0,3],["cm",0,0,3],["speed",0,0,3]],
    "tall":[["height",0,0,3],["meter",0,0,3],["M",0,0,3],["foot",0,0,3],["inch",0,0,3],["centimeter",0,0,3],["cm",0,0,3]],
    "far":[["kilometer",0,0,3],["KM",0,0,3],["meter",0,0,3],["M",0,0,3]],
    "distant":[["kilometer",0,0,3],["KM",0,0,3],["meter",0,0,3],["M",0,0,3]],
    "remote":[["kilometer",0,0,3],["KM",0,0,3],["meter",0,0,3],["M",0,0,3]],
    "wide":[["kilometer",0,0,3],["KM",0,0,3],["meter",0,0,3],["M",0,0,3],["foot",0,0,3],["inch",0,0,3],["centimeter",0,0,3],["cm",0,0,3]],
    "broad":[["kilometer",0,0,3],["KM",0,0,3],["meter",0,0,3],["M",0,0,3],["foot",0,0,3],["inch",0,0,3],["centimeter",0,0,3],["cm",0,0,3]],
    "deep":[["kilometer",0,0,3],["KM",0,0,3],["meter",0,0,3],["M",0,0,3],["foot",0,0,3],["inch",0,0,3],["centimeter",0,0,3],["cm",0,0,3]],
    "thick":[["meter",0,0,3],["M",0,0,3],["foot",0,0,3],["inch",0,0,3],["centimeter",0,0,3],["cm",0,0,3]],
    "close":[["kilometer",0,1,4],["KM",0,1,4],["meter",0,1,4],["M",0,1,4],["foot",0,1,4],["inch",0,1,4],["centimeter",0,1,4],["cm",0,1,4]],

    "vast":[["centiare",0,0,3],["square",0,0,3]],

    "huge":[["stere",0,0,3]],
    
    "hot":[["temperature",0,0,3],["centigrade",0,0,3],["centigrade",0,0,3],["celsius",0,0,3],["fahrenheit",0,0,3]],
    "warm":[["temperature",0,0,3],["centigrade",0,0,3],["centigrade",0,0,3],["celsius",0,0,3],["fahrenheit",0,0,3]],
    "cold":[["temperature",0,1,4],["centigrade",0,1,4],["centigrade",0,1,4],["celsius",0,1,4],["fahrenheit",0,1,4]],

    "rich":[["money",0,0,3],["pound",0,0,3],["gold",0,0,3],["dollar",0,0,3]],
    "expensive":[["money",0,0,3],["price",0,0,3],["cost",0,0,3],["pound",0,0,3],["gold",0,0,3],["dollar",0,0,3],["total",0,0,3]],
    "costly":[["money",0,0,3],["price",0,0,3],["cost",0,0,3],["pound",0,0,3],["gold",0,0,3],["dollar",0,0,3]],
    "cheap":[["money",0,1,4],["price",0,1,4],["cost",0,1,4],["pound",0,1,4],["gold",0,1,4],["dollar",0,1,4],["total",0,1,4]],
    
    "wet":[["humidness",0,0,3],["humidity",0,0,3]],
    "damp":[["humidness",0,0,3],["humidity",0,0,3]],

    "clever":[["intelligence",0,0,3],["wisdom",0,0,3],["wit",0,0,3],["IQ",0,0,3]],
    "wise":[["intelligence",0,0,3],["wisdom",0,0,3],["wit",0,0,3],["IQ",0,0,3]],
    "smart":[["intelligence",0,0,3],["wisdom",0,0,3],["wit",0,0,3],["IQ",0,0,3]],
    "foolish":[["intelligence",0,1,4],["wisdom",0,1,4],["wit",0,1,4],["IQ",0,1,4]],

    "loud":[["volume",0,0,3],["decibel",0,0,3]],
    "quiet":[["volume",0,1,4],["decibel",0,1,4]],
}


A_ADJ_WORD_DIRECTION = {
    "old":3,
    "young":4,
    "heavy":3,
    "fat":3,
    "light":4,
    "late":3,
    "early":4,
    "elderly":3,
    "long":3,
    "short":4,    
    "quick":4,
    "fast":4,
    "slow":3,
    "high":3,
    "tall":3,
    "far":3,
    "wide":3,
}






RELATED_WORD = {
    "DATE":["year","time","date","datetime","cal"],
    "YEAR":["year"],
    "TIME":["time","datetime","hour"],
}



ABSOLUTELY_GREATER_DICT = {
    # it will be combine to ABSOLUTELY_GRSM_DICT
    "above":3,
    "great":3,
    "large":3,
    "heavy":3,
    "after":3,
    "more":3,
    "high":3,
    "over":3,
    "most":3,
    # "large":3,
    "later":3,
    "big":3,
    "long":3,
    "exceed":3,
    ">":3,
    "<":4,

    "new":3,
    "great":3,
    "late":3,
    "far":3,
    "main":3,
    "major":3,
    "difficult":3,
    "strong":3,
    "hard":3,
    "wide":3,
    "serious":3,
    # "top":3,
    "hot":3,
    "deep":3,
    "primary":3,
    "huge":3,
    "rich":3,
    "powerful":3,
    "complex":3,
    "warm":3,
    "broad":3,
    "bright":3,
    "expensive":3,
    "dangerous":3,
    "fat":3,
    "obese":3,
    "thick":3,
    "fast":3,
    "elderly":3,
    "grand":3,
    "vast":3,
    "severe":3,
    "permanent":3,
    "sharp":3,
    "enormous":3,
    "tough":3,
    "extensive":3,
    "wet":3,
    "damp":3,
    "rapid":3,
    "fixed":3,
    "sweet":3,
    "rough":3,
    "advanced":3,
    "extreme":3,
    "favourite":3,
    "favorite":3,
    "widespread":3,
    "numerous":3,
    "remote":3,
    "distant":3,
    "clever":3,
    "wise":3,
    "frequent":3,
    "intense":3,
    "generous":3,
    "loud":3,
    "superb":3,
    "superior":3,
    "spectacular":3,
    "giant":3,
    "intensive":3,
    "steep":3,
    "excessive":3,
    "striking":3,
    "fierce":3,
    "precious":3,
    "smart":3,
    "endless":3,
    "super":3,
    "notable":3,
    "profound":3,
    "immense":3,
    "worthy":3,
    "redundant":3,
    "lengthy":3,
    "costly":3,
    "delicious":3,
    "grim":3,
    "rising":3,
    "formidable":3,
    "mighty":3,
    "eventual":3,

    "tall":3,
    "popular":3,
    "predominantly":3,
    "predominant":3,
    "dominant":3,
    "main":3,
    "major":3,
    "prime":3,
    "primary":3,
    "key":3,
    "furth":3,
    "further":3,
    "furthest":3,
    "good":3,
    "recent":3
}


ABSOLUTELY_SMALLER_DICT = {
    # it will be combine to ABSOLUTELY_GRSM_DICT
    "below":4,
    "light":4,
    "before":4,
    "less":4,
    "early":4,
    "earlier":4,
    "few":4,
    "small":4,
    "little":4,
    "low":4,
    "few":4,
    "least":4,
    "under":4,
    "minimum":4,
    "short":4,
    "rare":4,

    # "low":4,
    "easy":4,
    "close":4,
    "previous":4,
    "cold":4,
    "light":4,
    "cheap":4,
    "quick":4,
    "slow":4,
    "tiny":4,
    "dry":4,
    "thin":4,
    "ancient":4,
    "brief":4,
    "weak":4,
    "slight":4,
    "smooth":4,
    "tight":4,
    "faint":4,
    "lesser":4,
    "shallow":4,
    "vague":4,
    "reduced":4,
    "minimal":4,
    "insufficient":4,
    "slim":4,
    "prior":4,
    "foolish":4,

    "old":4,
    "bad":4,
    "quiet":4
}

COUNTRYS_DICT = {}
COUNTRYS = [
["United States of America","United States","USA","US","America","American","Washington"],
["Washington","WA"],
["People's Republic of China","China","Chinese","PRC","Beijing"],
["Beijing","BJ"],
["Japan","Cipango","Japanese","Nipponese","Tokyo"],
["Tokyo","TKY"],
["Germany","Deutschland","German","Germanic","Berlin"],
["Berlin","BL","BLN"],
["United Kingdom","England","UK","Great Britain","Britain","British","Britisher","Englishman","English","London"],#,"Briton"
["London","LDN"],
["France","French","Frenchman","Paris"],
["Paris","PAR"],
["India","Indian","New Delhi"],
["New Delhi","ND"],
["Italy","Italian","Rome"],
["Rome","ROM","RM"],
["Brazil","Brazilian","Brasilia"],
["Brasilia","OT"],
["Canada","Canadian","Ottawa"],
["Ottawa","OT"],
["Russia","Russian Federation","Russian","Moscow"],
["Moscow","MSK"],
["south Korea","Republic of Korea","South Korean","Corean","Seoul"],
["Seoul","Seo"],
["Spain","Spanish","Madrid"],
["Madrid","MAD"],
["Norway","Norseland","Norse","Norwegian","Norwegian","Oslo"],
["Oslo","OS"],
["Mexico","Mexican","Mex","Mexico City"],
["Mexico City","CDMX","MC"],
["Australia","Aussie","Australian","Canberra"],
["Canberra","CAN"],
["the Netherlands","Holland","Hollander","Netherlander","Dutch","Dutchman","Amsterdam"],
["Amsterdam","AMS"],
["Saudi Arabia","Kingdom of Saudi Arabia","Saudi Arabian","Saudi","Riyadh"],
["Riyadh","RY"],
["Turkey","Turk","Turco","Turkish","Osmanli","Ankara"],
["Ankara","ANK"],
["Switzerland","Swiss","Switzer","Helvetian","Bern"],
["Bern","BZ","BZB"],
["Sweden","Swede","Swedish","Stockholm"],
["Stockholm","SK","STHLM"],
["Indonesia","Republic of Indonesia","Indonesian","Bahasa Indonesia","Jakarta"],
["Jakarta","JK"],
["Belgium","The Kingdom of Belgium","Belgian","Brussels"],
["Brussels","BXL","BR"],
["Thailand","Kingdom of Thailand","Thai","Bangkok"],
["Bangkok","BKK","BK"],
["Argentina","Argentinean","Argentine","Buenos Aires"],
["Buenos Aires","BA","B.A.","Bs As"],
["Poland","Polish","Warsaw"],
["Warsaw","WS"],
["Ukraine","Ukrainian","Kyiv"],
["Kyiv","KV"],
["Iran","Iranian","Teheran"],
["Teheran","TEH"],
["Nigeria","Federal Republic of Nigeria","Nigerian","Abuja"],
["Abuja","ABV"],
["Venezuela","Bolivarian Republic of Venezuela","Venezuelian","Venezuelan","Caracas"],
["Caracas","CCS"],
["Ireland","Eire","The State of Israel","Irish","Irisher","Israelite","Dublin"],
["Dublin","DUB","DB"],
["Israel","The State of Israel","Israeli","Israelite","Jerusalem"],
["Jerusalem","JRS"],
["Denmark","The Kingdom of Denmark","Dane","Danish","Copenhagen"],
["Copenhagen","COP"],
["Malaysia","Malaysian","Malay","Kuala Lumpur"],
["Kuala Lumpur","KL","KUL"],
["Philippines","RP","Republic of the Philippines","Philippinese","Filipino","Philippine","Manila"],
["Manila","MN","MNL"],
["Pakistan","Pakistani","Islamabad"],
["Islamabad","ISB"],
["Chile","Chilean","chilian","Santiago"],
["Santiago","SAN","AGO"],
["The People's Republic of Bangladesh","Bangladesh","Bengali","Bengalese","Dhaka"],
["Dhaka","DAC","DHK"],
["Finland","Finnish","Finns","Helsinki"],
["Helsinki","HEL","HL"],
["Egypt","Egyptian","Cairo"],
["Cairo","CAI"],
["Czech Republic","Czech","Prague"],
["Prague","PR","PRG"],
["Vietnam","Viet Nam","Vietnamese","Ha Noi"],
["Ha Noi","HAN"],
["Rumania","Romania","Roumania","Romanian","Rumanian","Bucharest"],
["Bucharest","BUH"],
["Portugal","The Republic of Portugal","Portuguese","Lisbon"],
["Lisbon","LIS","LX"],
["Peru","The Republic of Peru","Peruvian","Lima"],
["Lima","LM"],
["Greece","Greek","Hellene","Argive","Grecian","Athens"],
["Athens","ATH","AS"],
["New Zealand","Nz","New Zealander","Kiwi","Zelanian","Wellington"],
["Wellington","WL","WEL"],
["Hungary","Hungarian","Budapest"],
["Budapest","BP"],
["korea","North Korea","koreans","Pyongyang"],
["Pyongyang","FNJ"],
["Slovak Republic","Slovakia","Slovak","Slovakian","Bratislava"],
["Bratislava","BV"],

["Sri Lanka","Sri Lankan"],
["Kenya","Kenyan"],
["Ethiopia","Ethiopian","Ethiopic"],
["Syria","The Syrian Arab Republic","Syrian"],
["Burma","Myanmar","Burmese"],
["Luxembourg","Luxemburger","Luxembourger"],
["Republic of Belarus","Byelorussian"],
["Lebanon","Lebanese"],
["Tanzania","Tanzanian"],
["Libya","Libyan"],
["Paraguay","Paraguayan"],
["Sudan","Sudanese"],
["Nepal","Nepalese","Nepali"],
["Yemen","Republic of Yemen","Yemenese","Yemeni","Yemenite"],
["Uganda","Ugandan"],
["Iceland","the Republic of Iceland","Icelandic","icelander"],
["Laos","Lao People's Republic","Lao"],
["Mauritius","The Republic of Mauritius","Mauritian"],
["Mongolia","People's Republic of Mongolia","Mongolian","Mongol"],

["Asia","Asian"],
["Europe","European","Euro"],
["Oceania","Oceanian"],
["African","Africa"],


["Alabama","AL"],# 50 states of america
["Alaska","AK"],
["Arizona","AZ"],
["Arkansas","AR"],
["California","CA"],
["Colorado","CO"],
["Connecticut","CT"],
["Delware","DE"],
["Florida","FL"],
["Georgia","GA"],
["Hawaii","HI"],
["Idaho","ID"],
["Illinois","IL"],
["Indiana","IN"],
["Iowa","IA"],
["Kansas","KS"],
["Kentucky","KY"],
# ["Louisiana","LA"],
["Los Angeles","Louisiana","LA"],
["Maine","ME"],
["Maryland","MD"],
["Massachusetts","MA"],
["Michigan","MI"],
["Minnesota","MN"],
["Mississippi","MS"],
["Montana","MT"],
["Nebraska","NE"],
["Nevada","NV"],
["New Hampshire","NH"],
["New Jersey","NJ"],
["New Mexico","NM"],
["New York","NY"],
["North Carolina","NC"],
["North Dakota","ND"],
["Ohio","OH"],
["Oklahoma","OK"],
["Oregon","OR"],
["Pennsylvania","PA"],
["Rhode Island","RI"],
["South Carolina","SC"],
["South Dakota","SD"],
["Tennessee","TN"],
["Texas","TX"],
["Utah","UT"],
["Vermont","VT"],
["Virginia","VA"],
["Washington","WA"],
["West Virgin","WV"],
["Wisconsin","WI"],
["Wyoming","WY"],
["Dist. Of Columbia","Columbia","DC"]
]

ALL_JJS={
"most":"most",
"biggest":"big",
"briefest":"brief",
"brightest":"bright",
"broadest":"broad",
"cheapest":"cheap",
"cleverest":"clever",
"closest":"close",
"coldest":"cold",
"dampest":"damp",
"deepest":"deep",
"driest":"dry",
"earliest":"early",
"easiest":"easy",
"extremest":"extreme",
"faintest":"faint",
"fastest":"fast",
"fattest":"fat",
"greatest":"great",
"grimmest":"grim",
"heaviest":"heavy",
"highest":"high",
"hottest":"hot",
"hugest":"huge",
"intensest":"intense",
"largest":"large",
"latest":"late",
"lengthiest":"lengthy",
"lightest":"light",
"littlest":"little",
"longest":"long",
"loudest":"loud",
"lowest":"low",
"mightiest":"mighty",
"newest":"new",
"shallowest":"shallow",
"sharpest":"sharp",
"shortest":"short",
"slightest":"slight",
"slimmest":"slim",
"slowest":"slow",
"smallest":"small",
"smartest":"smart",
"smoothest":"smooth",
"steepest":"steep",
"strongest":"strong",
"sweetest":"sweet",
"thickest":"thick",
"thinnest":"thin",
"tightest":"tight",
"tiniest":"tiny",
"toughest":"tough",
"vaguest":"vague",
"vastest":"vast",
"warmest":"warm",
"weakest":"weak",
"wettest":"wet",
"widest":"wide",
"wisest":"wise",
"worthiest":"worthy",
"tallest":"tall",
"oldest":"old",
"youngest":"young",
}


for i,c in enumerate(COUNTRYS):
    for j,cn in enumerate(c):
        COUNTRYS[i][j] = COUNTRYS[i][j].lower()
    for cnn in COUNTRYS[i]:
        COUNTRYS_DICT[cnn] = [c for c in COUNTRYS[i] if c != cnn]


SPECIAL_DB_WORD = ["GPE","ORG","PERSON"]


ABSOLUTELY_GREATER = ABSOLUTELY_GREATER_DICT.keys()
ABSOLUTELY_SMALLER = ABSOLUTELY_SMALLER_DICT.keys()
ABSOLUTELY_GRSM_DICT = copy.deepcopy(ABSOLUTELY_GREATER_DICT) 
ABSOLUTELY_GRSM_DICT.update(ABSOLUTELY_SMALLER_DICT)



def word_is_grsm(word_tok):
    if word_tok.lemma_ in ABSOLUTELY_GRSM_DICT:
        return True
    elif word_tok.lemma_ == word_tok.text and word_tok.text.endswith("er") and word_tok.text[:-2] in ABSOLUTELY_GRSM_DICT:
        return True
    return False




def table_match(table_str,schema):
    table_id = -1
    table_str = " ".join(table_str).split(" , ")
    return schema.table_match(table_str)

def clean_stop_word(col_str):
    if col_str:
        if " age " not in col_str:
            col_str = col_str.replace("youngest","youngest age")
            col_str = col_str.replace("oldest","oldest age")
        col_str = col_str.split(" ")
        for i in reversed(range(len(col_str))):
            if col_str[i] in STOP_WORDS and col_str[i] != "," or col_str[i].isdigit():
                del col_str[i]
        return col_str
    return None

def column_match(col_str,schema,table_str = None,table_id = -1):
    col_names = " ".join(col_str).split(" , ")
    agg_id, col_id = schema.column_match(table_id,col_names)
    if table_id < 0 and not col_id and table_str and len(table_str) == 1 and len(col_names)==1:
        agg_id, col_id = schema.column_match(table_id,[col_names[0] + " " +table_str[0]]) # for suitable for number of column name, I modify the table name as behind
    return agg_id, col_id, table_id


