CLAUSE_KEYWORDS = ('select', 'from', 'where', 'group', 'order', 'limit', 'intersect', 'union', 'except')
JOIN_KEYWORDS = ('join', 'on', 'as')

WHERE_OPS = ('not', 'between', '=', '>', '<', '>=', '<=', '!=', 'in', 'like', 'is', 'exists', 'not in', 'not like', 'not between', 'join')
UNIT_OPS = ('none', '-', '+', "*", '/')
AGG_OPS = ('none', 'max', 'min', 'count', 'sum', 'avg')
TABLE_TYPE = {
    'sql': "sql",
    'table_unit': "table_unit",
}

AND_OR_OPS = ('and', 'or')
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









# The following unit back only for the original table file.


def num_col_unit_back(col_unit,tables_json):
    str_ = ""
    if col_unit[0]:
        str_ += AGG_OPS[col_unit[0]] + " ( "
    if col_unit[2]:
        str_ += " distinct "
    if isinstance(col_unit[1],str):
        str_ +=  col_unit[1]
    elif col_unit[1] > 0: # col idx
        str_ +=  tables_json["table_names_original"][tables_json["column_names_original"][col_unit[1]][0]]+"."+tables_json["column_names_original"][col_unit[1]][1]
    elif tables_json['column_types'].count("others") >= 2:
        cc = 0
        for col in tables_json['column_names']:
            if col[1] == "*":
                cc += 1
                if cc > 1:
                    break
        if cc > 1:
            str_ +=  tables_json["table_names_original"][tables_json["column_names_original"][col_unit[1]][0]]+"."+tables_json["column_names_original"][col_unit[1]][1]
        else:
            str_ += "*"
    else:
        str_ += "*"
    
    # str_ += tables_json["table_names_original"][tables_json["column_names_original"][col_unit[1]][0]]+"."+tables_json["column_names_original"][col_unit[1]][1]
    if col_unit[0]:
        str_ +=  " ) "
    return str_


def num_val_unit_back(val_unit,tables_json):
    if val_unit[0] and (type(val_unit[2]) == tuple or type(val_unit[2]) == list):
        return num_col_unit_back(val_unit[1],tables_json) + " " + UNIT_OPS[val_unit[0]]  + " " +  num_col_unit_back(val_unit[2],tables_json)
    return num_col_unit_back(val_unit[1],tables_json)


def select_col_num_val_back(select_col,tables_json):
    """
    tables_json is original table. not new table type.
    """
    val = ""
    if select_col[0] > 0: # agg
        val += AGG_OPS[select_col[0]] + ' ( '

    val += num_val_unit_back(select_col[1], tables_json)

    if select_col[0] > 0:
        val += ' ) '
        
    return val


def select_string_back_based_idx(selec,tables_json):
    """
    tables_json is original table. not new table type.
    """
    if selec:
        re = "select "
        if selec[0]:
            re += " distinct "
        for val in selec[1]:
            re += select_col_num_val_back(val,tables_json) + ' , '
        return re[:-3]
    return ""

def from_string_back_based_idx(from_, tables_json):
    re = ' from '
    if len(from_['table_units']) == 1 and  from_['table_units'][0][0] == "sql" and len(from_['table_units'][0]) == 2 :
        return " from ( " +  sql_back(from_['table_units'][0][1], tables_json) + " )"
    elif not from_['conds'] and len(from_['table_units']) > 1:
        for t in from_['table_units']:
            re += tables_json["table_names_original"][t[1]] + " join "
        re = re[:-5]
    elif not from_['conds']:
        if isinstance(from_['table_units'][0][1],str):
            re += from_['table_units'][0][1]
        else:
            re += tables_json["table_names_original"][from_['table_units'][0][1]] + " "
    else:
        tables_list = set()
        for conds in from_['conds']:
            if not isinstance(conds,str):
                con_str = condition_back_based_idx(conds,tables_json)
                t1 = tables_json['column_names'][conds[2][1][1]][0]
                t2 = tables_json['column_names'][conds[3][1]][0]
                if t1 not in tables_list and t2 not in tables_list:
                    if t1 == t2 and not tables_list and len(from_['table_units']) == 2:
                        if from_['table_units'][0][1] != t1:
                            re += tables_json['table_names_original'][from_['table_units'][0][1]] + " "
                            tables_list.add(from_['table_units'][0][1])
                        else:
                            re += tables_json['table_names_original'][from_['table_units'][1][1]] + " "
                            tables_list.add(from_['table_units'][1][1])
                    else:
                        re += tables_json['table_names_original'][t1] + " "
                    re += " join " + tables_json['table_names_original'][t2] + " on "
                    re += con_str
                    tables_list.add(t1)
                    tables_list.add(t2)
                else:
                    if t1 not in tables_list:
                        re += " join " + tables_json['table_names_original'][t1] + " on "
                        tables_list.add(t1)
                    elif t2 not in tables_list:
                        re += " join " + tables_json['table_names_original'][t2] + " on "
                        tables_list.add(t2)
                    else:
                        re += " and "
                    re += con_str
        for t in from_['table_units']:
            if t[1] not in tables_list:
                re += " join " + tables_json['table_names_original'][t[1]] 
    return re



def orderby_string_back_based_idx(val_unit,limit,tables_json):
    """
    tables_json is original table. not new table type.
    """
    re_limit = ""
    if limit:
        re_limit += " limit " + str(limit)

    if not val_unit:
        return "" + re_limit
    re = " order by "
    for val in val_unit[1]:
        re += num_val_unit_back(val,tables_json) + " , "
         
    return re[:-3] + " " + val_unit[0] + re_limit

def groupby_string_back_based_idx(col_units,tables_json):
    """
    tables_json is original table. not new table type.
    """
    re_group = ""
    if col_units:
        re_group += " group by "
    else:
        return ""
    for col in col_units:
        re_group += num_col_unit_back(col,tables_json) + " , "
    return re_group[:-2]

def having_string_back_based_idx(having_unit,tables_json):
    """
    tables_json is original table. not new table type.
    """
    re_where= ""
    if having_unit:
        re_where += " having "
    else:
        return ""
    for where in having_unit:
        if isinstance(where,str):
            re_where += " " + where + " "
        else:
            re_where += condition_back_based_idx(where,tables_json)
    return re_where

def condition_back_based_idx(conds,tables_json):
    re_conds = ""
    re_conds += num_val_unit_back(conds[2],tables_json)
    if conds[0] == 2:
        re_conds += " enot "
    elif conds[0]:
        re_conds += " not "
    re_conds += " " + WHERE_OPS[conds[1]] + " "
    if isinstance(conds[3],dict):
        re_conds += ' ( '
        re_conds += sql_back(conds[3],tables_json)
        re_conds += ' ) '
    elif isinstance(conds[3],tuple) or isinstance(conds[3],list):
        re_conds += num_col_unit_back(conds[3],tables_json)
    else:
        re_conds += str(conds[3])
    if conds[4]:
        re_conds += " and "
        re_conds += str(conds[4])
    return " " + re_conds + " "

def where_string_back_based_idx(where_unit,tables_json):
    """
    tables_json is original table. not new table type.
    """
    re_where= ""
    if where_unit:
        re_where += " where "
    else:
        return ""
    for where in where_unit:
        if isinstance(where,str):
            re_where += " " + where + " "
        else:
            re_where += condition_back_based_idx(where,tables_json)
    return re_where


def group_string_back_based_idx(group, tables_json):
    re = ""
    if group:
        re += " group by "
        for g in group:
            re += tables_json['table_names_original'][tables_json['column_names_original'][g[1]][0]] + "." + tables_json['column_names_original'][g[1]][1] + " , "
        re = re[:-2]
    return re

def sql_back(sql, table):
    select = select_string_back_based_idx(sql["select"],table)
    from_ = ""
    if "from" in sql:
        from_ =  from_string_back_based_idx(sql["from"],table)
    where = where_string_back_based_idx(sql["where"],table)
    group = group_string_back_based_idx(sql["groupBy"],table)
    having = having_string_back_based_idx(sql["having"],table)
    order = orderby_string_back_based_idx(sql["orderBy"],sql["limit"],table)
    sql_str = select + from_ + where + group +  having + order
    if sql['intersect']:
        if type(sql['intersect']) == dict:
            sql_str = sql_str + " intersect " + sql_back(sql['intersect'], table)
        else:
            sql_str = sql_str + " intersect select" + select_col_num_val_back(sql['intersect'], table)
    if sql['union']:
        if type(sql['union']) == dict:
            sql_str = sql_str + " union " + sql_back(sql['union'], table)
        else:
            sql_str = sql_str + " union select" + select_col_num_val_back(sql['union'], table)
    if sql['except']:
        sql_str = sql_str + " except " + sql_back(sql['except'], table)
    return sql_str











def col_unit_contain_agg(col_unit,tables_with_alias = None):
    if col_unit == None:
        return False
    if col_unit[0]  > 0:
        return True
    return False



def val_unit_contain_agg(val_unit,tables_with_alias = None):
    return col_unit_contain_agg(val_unit[1])



def return_all_select_col(select):
    """
    select:
        sql['select']
    """
    re_select = []
    distinct = select[0]
    re_agg = []
    for column in select[1]:
        re_select.append(column[1][1][1])
        re_agg.append(column[0])
    return distinct,re_agg,re_select


def return_all_orderby_col(orderby):
    """
    select:
        sql['orderBy']
    """
    if not orderby:
        return [],None
    re_orderby = []
    for col in orderby[1]:
        re_orderby.append(col[1][1])
    return re_orderby, orderby[0]


def return_all_where_col(where):
    """
    where:
        sql['where']
    """
    re_where = []
    re_where_format = []
    for column in where:
        one_where = []
        if column not in ALL_COND_OPS:   
            re_where.append(column[2][1][1])
            where_right = column[3]
            if type(where_right) == str or type(where_right) == int or type(where_right) == float or type(where_right) == dict:
                one_where = [column[2][1][1],column[1],"value"]
                pass
            else:
                re_where.append(column[3][1])
                one_where = [column[2][1][1],column[1],column[3][1]]
            re_where_format.append(one_where)
    return re_where,re_where_format


def replace_the_second(sql):
    if sql.count("select ") > 1:
        idx = sql.index("select ")

def cut_sql_to_piece(sql):
    sql = sql.lower()
    cut_order = [" select "," from "," where "," group "," having "," order "]
    dict_ ={}
    idx_next = 1
    idx_now = 0
    while True:
        if idx_next >= len(cut_order):
            dict_[cut_order[idx_now].strip()] = sql
            break
        
        if cut_order[idx_next] in sql:
            split = sql.split(cut_order[idx_next])
            dict_[cut_order[idx_now].strip()] = split[0]
            sql = cut_order[idx_next] + split[1]
            idx_now = idx_next
        idx_next += 1 
    return dict_
