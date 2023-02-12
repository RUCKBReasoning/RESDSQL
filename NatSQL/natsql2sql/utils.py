import re

CLAUSE_KEYWORDS = ('select', 'from', 'where', 'group', 'order', 'limit', 'intersect', 'union', 'except')
JOIN_KEYWORDS = ('join', 'on', 'as')

WHERE_OPS = ('not', 'between', '=', '>', '<', '>=', '<=', '!=', 'in', 'like', 'is', 'exists', 'not in', 'not like')

UNIT_OPS = ('none', '-', '+', "*", '/')
AGG_OPS = ('none', 'max', 'min', 'count', 'sum', 'avg')
TABLE_TYPE = {
    'sql': "sql",
    'table_unit': "table_unit",
}

AND_OR_OPS = ('and', 'or', 'except_', 'intersect_', 'union_', 'sub')#COND_OPS = ('and', 'or')
ALL_COND_OPS = ('and', 'or', 'except_', 'intersect_', 'union_', 'sub')
SPECIAL_COND_OPS = ('except_', 'intersect_', 'union_', 'sub')

SQL_OPS = ('intersect', 'union', 'except')
ORDER_OPS = ('desc', 'asc')



def col_unit_back(col_unit,tables_with_alias = None):
    """
        tables_with_alias is needed when col_unit is T1.column not table.column
    """
    if col_unit == None:
        return None

    bool_agg = False
    col = ""

    if col_unit[2]  != None and col_unit[2]:
        col = " distinct "
    if col_unit[0]  > 0:
        col = AGG_OPS[col_unit[0]] + '(' + col
        bool_agg = True

    name = col_unit[1]
    if name.endswith("__"):
        name = name[:-2]
    if name.startswith("__"):
        name = name[2:]
    if name == 'all':
        name = '*'
    if name.endswith("*"):
        name = '*'

    nameArray = name.split('.')
    if len(nameArray) == 2 and tables_with_alias:
        table_name = nameArray[0]
        for key,value in tables_with_alias.items():
            if key != table_name and value == table_name:
                name = key + "." + nameArray[1]
                break

    col = col + name

    if bool_agg:
        col = col + ')'

    return col


def val_unit_back(val_unit,tables_with_alias = None):

    val = ""

    col_1 = col_unit_back(val_unit[1], tables_with_alias)
    col_2 = col_unit_back(val_unit[2], tables_with_alias) 

    if val_unit[0] > 0 and col_2 != None: 
        val = val + col_1 +" "+ UNIT_OPS[val_unit[0]]+" "+col_2
    else:
        val = val + col_1

    return val


def select_unit_back(val_unit,tables_with_alias = None):
    val = ""
    if val_unit[0] > 0: # agg
        val = AGG_OPS[val_unit[0]] + '('

    val += val_unit_back(val_unit[1], tables_with_alias)

    if val_unit[0] > 0:
        val = val + ')'
        
    return val




def col_unit_contain_agg(col_unit,tables_with_alias = None):
    if col_unit == None:
        return False
    if col_unit[0]  > 0:
        return True
    return False



def val_unit_contain_agg(val_unit,tables_with_alias = None):
    return col_unit_contain_agg(val_unit[1])





def is_float(s):
    s = str(s)
    if s.count('.') ==1:
        left = s.split('.')[0]
        right = s.split('.')[1]
        if right.isdigit():
            if left.count('-')==1 and left.startswith('-'):
                num = left.split['-'][-1]
                if num.isdigit():
                    return True
            elif left.isdigit():
                return True
    return False

def is_negative_digit(s):
    s = str(s)
    if s.startswith('-') and len(s) > 1:
        return s[1:].isdigit()
    return False

NUM = {"zero":0,"single":1,"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,"nine":9,"ten":10,"eleven":11,"twelve":12,"thirteen":13,"fourteen":14,"fifteen":15,"sixteen":16,"seventeen":17,"eighteen":18,"nineteen":19,"twenty":20,"once":1,"twice":2\
    ,"first":1,"second":2,"third":3,"fourth":4,"fifth":5,"sixth":6,"seventh":7,"eighth":8,"ninth":9,"tenth":10}

def str_is_num(s):
    return s.lower() in NUM.keys() or s.replace(",",'').isdigit() or is_float(s.replace(",",'')) or is_negative_digit(s.replace(",",'')) 


def str2num(s):
    if s.startswith("'") and s.endswith("'") and len(s) > 2:
        s = s[1:-1]
    elif s.startswith('"') and s.endswith('"') and len(s) > 2:
        s = s[1:-1]
    if s.lower() in NUM.keys():
        return NUM[s.lower()]
    elif s.replace(",",'').isdigit():
        return int(s.replace(",",''))
    elif is_float(s):
        return float(s)
    elif is_negative_digit(s.replace(",",'')):
        return int(s.replace(",",''))
    return 0


def str_is_date(s):
    if re.fullmatch(r"^[A-Za-z]+$",s, flags=0):
        return False
    elif re.fullmatch(r'((\d{4}((_|-|/){1}\d{1,2}){2})|(\d{1,2}(_|-|/)){2}\d{4}){0,1}\s{0,1}(\d{2}(:\d{2}){1,2}){0,1}',s, flags=0):
        return True
    elif re.fullmatch(r'(\d{1,2}(st|nd|rd|th){0,1}(,|\s)){0,1}((J|j)an(uary){0,1}|(F|f)eb(ruary){0,1}|(M|m)ar(ch){0,1}|(A|a)pr(il){0,1}|(M|m)ay|(J|j)un(e){0,1}|(J|j)ul(y){0,1}|(A|a)ug(ust){0,1}|(S|s)ep(tember){0,1}|(O|o)ct(ober){0,1}|(N|n)ov(ember){0,1}|(D|d)ec(ember){0,1})(\s|,)(\d{1,2}(st|nd|rd|th){0,1}(\s|,){1,3}){0,1}\d{4}',s, flags=0):
        return True
    return False


def str_is_special_num(s):
    if len(s) < 4 or not s.isdigit():
        return False
    elif re.fullmatch(r"^([1][5-9]\d{2}|[2][0]\d{2})$",s, flags=0):
        return False
    elif s.endswith("00"):
        return False
    else:
        return True