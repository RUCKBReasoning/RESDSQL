# Adapted from
# https://github.com/taoyds/spider/blob/master/process_sql.py


from .natsql2sql import inference_sql,Args,natsql_version as n_version
import sqlite3
import sys,os,copy
from nltk import word_tokenize
from .process_sql import get_tables_with_alias, Schema, get_schema
from .process_sql import parse_sql as parse_sql_original
from .process_sql import tokenize as tokenize_original
from .preprocess.sql_back import sql_back

CLAUSE_KEYWORDS = ('select', 'from', 'where', 'group', 'order', 'limit', 'intersect', 'union', 'except', 'distinct')
CLAUSE_KEYWORDS2 = ('select', 'from', 'where', 'group', 'order', 'limit', 'intersect', 'union', 'except')
JOIN_KEYWORDS = ('join', 'on', 'as')

WHERE_OPS = ('not', 'between', '=', '>', '<', '>=', '<=', '!=', 'in', 'like', 'is', 'exists', 'not in', 'not like', 'not like', 'join')
UNIT_OPS = ('none', '-', '+', "*", '/')
AGG_OPS = ('none', 'max', 'min', 'count', 'sum', 'avg')
TABLE_TYPE = {
    'sql': "sql",
    'table_unit': "table_unit",
}

COND_OPS = ('and', 'or', 'except_', 'intersect_', 'union_', 'sub')#COND_OPS = ('and', 'or')
SQL_OPS = ('intersect', 'union', 'except')
ORDER_OPS = ('desc', 'asc')

def natsql_version():
    return n_version()

class Schema_Star(Schema):
    def __init__(self, schema):
        Schema.__init__(self, schema)

    def _map(self, schema):
        idMap = {}#{'*': "__all__"}
        for key, vals in schema.iteritems() if sys.version_info < (3, 0) else schema.items():
            for val in vals:
                idMap[key.lower() + '.' + val.lower()] = key.lower() + '.' + val.lower()
            idMap[key.lower() + '.*'] = key.lower() + '.*'

        for key in schema:
            idMap[key.lower()] =  key.lower() 

        return idMap



class Schema_Num:
    """
    Simple schema which maps table&column to a unique identifier
    """
    def __init__(self, schema, table_json):
        self._schema = schema
        self._idMap = self._map(self._schema, table_json)

    @property
    def schema(self):
        return self._schema

    @property
    def idMap(self):
        return self._idMap

    def _map(self, schema, table_json):
        idMap = {'*': -1}    # old idx:-1; new idx: check table name
        for key, vals in schema.iteritems() if sys.version_info < (3, 0) else schema.items() :
            for val in vals:
                target = key.lower() + "." + val.lower()
                for idx, name in enumerate(table_json['table_column_names_original']):
                    if name[1].lower() == target:
                        idMap[key.lower() + "." + val.lower()] = table_json['link_back'][idx][0] # new idx:0; old idx:1
                        break
            target = key.lower() + '.*'
            for idx, name in enumerate(table_json['table_column_names_original']):
                if name[1].lower() == target:
                    idMap[target] = table_json['link_back'][idx][0] # new idx:0; old idx:1
                    break

        for key in schema:
            idx = 0
            for name in table_json['table_names_original']:
                if name.lower() == key:
                    idMap[key.lower()] = idx
                    break
                idx += 1

        return idMap


def tokenize(string):
    string = str(string)
    string = string.replace("\'", "\"")  # ensures all string values wrapped by "" problem??
    quote_idxs = [idx for idx, char in enumerate(string) if char == '"']
    assert len(quote_idxs) % 2 == 0, "Unexpected quote"

    # keep string value as token
    vals = {}
    for i in range(len(quote_idxs)-1, -1, -2):
        qidx1 = quote_idxs[i-1]
        qidx2 = quote_idxs[i]
        val = string[qidx1: qidx2+1]
        key = "__val_{}_{}__".format(qidx1, qidx2)
        string = string[:qidx1] + key + string[qidx2+1:]
        vals[key] = val

    toks = [word.lower() for word in word_tokenize(string)]
    # replace with string value token
    for i in range(len(toks)):
        if toks[i] in vals:
            toks[i] = vals[toks[i]]

    # find if there exists !=, >=, <=
    eq_idxs = [idx for idx, tok in enumerate(toks) if tok == "=" or tok == "in" or tok == "like"]# make 'not in' and 'not like' together 
    eq_idxs.reverse()
    prefix = ('!', '>', '<')
    for eq_idx in eq_idxs:
        pre_tok = toks[eq_idx-1]
        if pre_tok in prefix:
            toks = toks[:eq_idx-1] + [pre_tok + "="] + toks[eq_idx+1: ]
        elif pre_tok == 'not' and toks[eq_idx].lower() in ["in","like"]:
            toks = toks[:eq_idx-1] + [pre_tok + " " + toks[eq_idx]] + toks[eq_idx+1: ]# make 'not in' and 'not like' together 
    return toks


def tokenize_nSQL(nsql, star_name, sepearte_star_name = True):
    nsql = nsql.replace("@.@","@")
    toks = tokenize(nsql)
    idx_star = 0
    remove_idx = []
    for idx,tok in enumerate(toks):
        if tok in['except', 'intersect', 'union']:
            toks[idx] = toks[idx] + '_'
        if tok == "@": #or tok == "@@@":
            toks[idx] = '@.@'
        if sepearte_star_name:
            if tok == '*':
                toks[idx] = star_name[idx_star].lower() + ".*"
                idx_star += 1
        else:
            if tok == '*':
                toks[idx] = toks[idx-2] + toks[idx-1] + toks[idx]
                remove_idx.append(idx-2)
                remove_idx.append(idx-1)
    remove_offset = 0
    for remove in remove_idx:
        del toks[remove - remove_offset]
        remove_offset += 1
    return toks



def parse_col(toks, start_idx, tables_with_alias, schema, default_tables=None):
    """
        :returns next idx, column id
    """
    tok = toks[start_idx]
    if tok == "*":
        return start_idx + 1, schema.idMap[tok]
    
    if tok == "@.@":
        return start_idx + 1, "@.@"

    if '.' in tok:  # if token is a composite
        alias, col = tok.split('.')
        key = tables_with_alias[alias] + "." + col
        if key not in schema.idMap and toks[start_idx+1] == "(":
                new_key = key
                for i in range(start_idx+1,len(toks)):
                    new_key += toks[i]
                    if new_key in schema.idMap:
                        return i+1,  schema.idMap[new_key]
        return start_idx+1, schema.idMap[key]

    assert default_tables is not None and len(default_tables) > 0, "Default tables should not be None or empty"

    for alias in default_tables:
        table = tables_with_alias[alias]
        if tok in schema.schema[table]:
            key = table + "." + tok
            if key not in schema.idMap and toks[start_idx+1] == "(":
                new_key = key
                for i in range(start_idx+1,len(toks)):
                    new_key += toks[i]
                    if new_key in schema.idMap:
                        return i+1,  schema.idMap[new_key]

            return start_idx+1, schema.idMap[key]

    assert False, "Error col: {}".format(tok)


def parse_col_unit(toks, start_idx, tables_with_alias, schema, default_tables=None):
    """
        :returns next idx, (agg_op id, col_id)
    """
    idx = start_idx
    len_ = len(toks)
    isBlock = False
    isDistinct = False
    if toks[idx] == '(':
        isBlock = True
        idx += 1

    if toks[idx] in AGG_OPS:
        agg_id = AGG_OPS.index(toks[idx])
        idx += 1
        assert idx < len_ and toks[idx] == '('
        idx += 1
        if toks[idx] == "distinct":
            idx += 1
            isDistinct = True
        idx, col_id = parse_col(toks, idx, tables_with_alias, schema, default_tables)
        assert idx < len_ and toks[idx] == ')'
        idx += 1
        return idx, [agg_id, col_id, isDistinct]

    if toks[idx] == "distinct":
        idx += 1
        isDistinct = True
    agg_id = AGG_OPS.index("none")
    idx, col_id = parse_col(toks, idx, tables_with_alias, schema, default_tables)

    if isBlock:
        assert toks[idx] == ')'
        idx += 1  # skip ')'

    return idx, [agg_id, col_id, isDistinct]


def parse_val_unit(toks, start_idx, tables_with_alias, schema, default_tables=None):
    idx = start_idx
    len_ = len(toks)
    isBlock = False
    if toks[idx] == '(':
        isBlock = True
        idx += 1

    col_unit1 = None
    col_unit2 = None
    unit_op = UNIT_OPS.index('none')

    idx, col_unit1 = parse_col_unit(toks, idx, tables_with_alias, schema, default_tables)
    if idx < len_ and toks[idx] in UNIT_OPS:
        unit_op = UNIT_OPS.index(toks[idx])
        idx += 1
        idx, col_unit2 = parse_col_unit(toks, idx, tables_with_alias, schema, default_tables)

    if isBlock:
        assert toks[idx] == ')'
        idx += 1  # skip ')'

    return idx, [unit_op, col_unit1, col_unit2]


def parse_table_unit(toks, start_idx, tables_with_alias, schema):
    """
        :returns next idx, table id, table name
    """
    idx = start_idx
    len_ = len(toks)
    key = tables_with_alias[toks[idx]]

    if idx + 1 < len_ and toks[idx+1] == "as":
        idx += 3
    else:
        idx += 1

    return idx, schema.idMap[key], key


def parse_value(toks, start_idx, tables_with_alias, schema, default_tables=None):
    idx = start_idx
    len_ = len(toks)

    isBlock = False
    if toks[idx] == '(':
        isBlock = True
        idx += 1

    if toks[idx] == 'select':
        idx, val = parse_sql(toks, idx, tables_with_alias, schema)
    elif "\"" in toks[idx]:  # token is a string value
        val = toks[idx]
        idx += 1
    elif "value" == toks[idx]:  # token is a string value
        val = 1
        idx += 1
    else:
        try:
            val = float(toks[idx])
            if str(val) != toks[idx]:
                val = int(toks[idx])
            idx += 1
        except:
            end_idx = idx
            while end_idx < len_ and toks[end_idx] != ',' and toks[end_idx] != ')'\
                and toks[end_idx] != 'join' and toks[end_idx] not in CLAUSE_KEYWORDS2 and toks[end_idx] not in COND_OPS:#toks[end_idx] != 'sub' and toks[end_idx] != 'or':
                end_idx += 1

            if toks[start_idx] in AGG_OPS and toks[end_idx] == ')': # Add for where column = agg(column)
                end_idx += 1

            idx, val = parse_col_unit(toks[start_idx: end_idx], 0, tables_with_alias, schema, default_tables)
            idx = end_idx

    if isBlock:
        assert toks[idx] == ')'
        idx += 1

    return idx, val


def parse_condition(toks, start_idx, tables_with_alias, schema, default_tables=None):
    idx = start_idx
    len_ = len(toks)
    conds = []

    if toks[idx] in COND_OPS:
        conds.append(toks[idx])
        idx += 1  # skip and/or

    while idx < len_:
        idx, val_unit = parse_val_unit(toks, idx, tables_with_alias, schema, default_tables)
        not_op = False
        if toks[idx] == 'not':
            not_op = True
            idx += 1

        assert idx < len_ and toks[idx] in WHERE_OPS, "Error condition: idx: {}, tok: {}".format(idx, toks[idx])
        op_id = WHERE_OPS.index(toks[idx])
        idx += 1
        val1 = val2 = None
        if op_id == WHERE_OPS.index('between'):  # between..and... special case: dual values
            idx, val1 = parse_value(toks, idx, tables_with_alias, schema, default_tables)
            assert toks[idx] == 'and'
            idx += 1
            idx, val2 = parse_value(toks, idx, tables_with_alias, schema, default_tables)
        else:  # normal case: single value
            idx, val1 = parse_value(toks, idx, tables_with_alias, schema, default_tables)
            val2 = None

        conds.append([not_op, op_id, val_unit, val1, val2])

        if idx < len_ and (toks[idx] in CLAUSE_KEYWORDS or toks[idx] in (")", ";") or toks[idx]=="join"):
            break

        if idx < len_ and toks[idx] in COND_OPS:
            conds.append(toks[idx])
            idx += 1  # skip and/or

    return idx, conds


def parse_select(toks, start_idx, tables_with_alias, schema, default_tables=None):
    idx = start_idx
    len_ = len(toks)

    assert toks[idx] == 'select', "'select' not found"
    idx += 1
    isDistinct = False
    if idx < len_ and toks[idx] == 'distinct':
        idx += 1
        isDistinct = True
    val_units = []

    while idx < len_ and toks[idx] not in CLAUSE_KEYWORDS:
        agg_id = AGG_OPS.index("none")
        if toks[idx] in AGG_OPS:
            agg_id = AGG_OPS.index(toks[idx])
            idx += 1
        idx, val_unit = parse_val_unit(toks, idx, tables_with_alias, schema, default_tables)
        val_units.append([agg_id, val_unit])
        if idx < len_ and toks[idx] == ',':
            idx += 1  # skip ','

    return idx, [isDistinct, val_units]


def parse_from(toks, start_idx, tables_with_alias, schema):
    """
    Assume in the from clause, all table units are combined with join
    """
    assert 'from' in toks[start_idx:], "'from' not found"

    len_ = len(toks)
    idx = toks.index('from', start_idx) + 1
    default_tables = []
    table_units = []
    conds = []

    while idx < len_:
        isBlock = False
        if toks[idx] == '(':
            isBlock = True
            idx += 1

        if toks[idx] == 'select':
            idx, sql = parse_sql(toks, idx, tables_with_alias, schema)
            table_units.append([TABLE_TYPE['sql'], sql])
        else:
            if idx < len_ and toks[idx] == 'join':
                idx += 1  # skip join
            idx, table_unit, table_name = parse_table_unit(toks, idx, tables_with_alias, schema)
            table_units.append([TABLE_TYPE['table_unit'],table_unit])
            default_tables.append(table_name)
        if idx < len_ and toks[idx] == 'join':
            idx += 1  # skip join
        if idx < len_ and toks[idx] == "on":
            idx += 1  # skip on
            idx, this_conds = parse_condition(toks, idx, tables_with_alias, schema, default_tables)
            conds.extend(this_conds)

        if isBlock:
            assert toks[idx] == ')'
            idx += 1
        if idx < len_ and (toks[idx] in CLAUSE_KEYWORDS or toks[idx] in (")", ";")):
            break

    return idx, table_units, conds, default_tables


def parse_where(toks, start_idx, tables_with_alias, schema, default_tables):
    idx = start_idx
    len_ = len(toks)

    if idx >= len_ or toks[idx] != 'where':
        return idx, []

    idx += 1
    idx, conds = parse_condition(toks, idx, tables_with_alias, schema, default_tables)
    return idx, conds


def parse_group_by(toks, start_idx, tables_with_alias, schema, default_tables):
    idx = start_idx
    len_ = len(toks)
    col_units = []

    if idx >= len_ or toks[idx] != 'group':
        return idx, col_units

    idx += 1
    assert toks[idx] == 'by'
    idx += 1

    while idx < len_ and not (toks[idx] in CLAUSE_KEYWORDS or toks[idx] in (")", ";")):
        idx, col_unit = parse_col_unit(toks, idx, tables_with_alias, schema, default_tables)
        col_units.append(col_unit)
        if idx < len_ and toks[idx] == ',':
            idx += 1  # skip ','
        else:
            break

    return idx, col_units


def parse_order_by(toks, start_idx, tables_with_alias, schema, default_tables):
    idx = start_idx
    len_ = len(toks)
    val_units = []
    order_type = 'asc' # default type is 'asc'

    if idx >= len_ or toks[idx] != 'order':
        return idx, val_units

    idx += 1
    assert toks[idx] == 'by'
    idx += 1

    while idx < len_ and not (toks[idx] in CLAUSE_KEYWORDS2 or toks[idx] in (")", ";")):
        idx, val_unit = parse_val_unit(toks, idx, tables_with_alias, schema, default_tables)
        val_units.append(val_unit)
        if idx < len_ and toks[idx] in ORDER_OPS:
            order_type = toks[idx]
            idx += 1
        if idx < len_ and toks[idx] == ',':
            idx += 1  # skip ','
        else:
            break

    return idx, [order_type, val_units]


def parse_having(toks, start_idx, tables_with_alias, schema, default_tables):
    idx = start_idx
    len_ = len(toks)

    if idx >= len_ or toks[idx] != 'having':
        return idx, []

    idx += 1
    idx, conds = parse_condition(toks, idx, tables_with_alias, schema, default_tables)
    return idx, conds


def parse_limit(toks, start_idx):
    idx = start_idx
    len_ = len(toks)

    if idx < len_ and toks[idx] == 'limit':
        idx += 2
        return idx, int(toks[idx-1])

    return idx, None


def parse_sql(toks, start_idx, tables_with_alias, schema):
    isBlock = False # indicate whether this is a block of sql/sub-sql
    len_ = len(toks)
    idx = start_idx

    sql = {}
    if toks[idx] == '(':
        isBlock = True
        idx += 1

    # parse from clause in order to get default tables
    from_end_idx, table_units, conds, default_tables = parse_from(toks, start_idx, tables_with_alias, schema)
    sql['from'] = {'table_units': table_units, 'conds': conds}
    # select clause
    _, select_col_units = parse_select(toks, start_idx, tables_with_alias, schema, default_tables)
    idx = from_end_idx
    sql['select'] = select_col_units
    # where clause
    idx, where_conds = parse_where(toks, idx, tables_with_alias, schema, default_tables)
    sql['where'] = where_conds
    # group by clause
    idx, group_col_units = parse_group_by(toks, idx, tables_with_alias, schema, default_tables)
    sql['groupBy'] = group_col_units
    # order by clause
    idx, order_col_units = parse_order_by(toks, idx, tables_with_alias, schema, default_tables)
    sql['orderBy'] = order_col_units
    # having clause
    idx, having_conds = parse_having(toks, idx, tables_with_alias, schema, default_tables)
    sql['having'] = having_conds
    # limit clause
    idx, limit_val = parse_limit(toks, idx)
    sql['limit'] = limit_val

    if isBlock:
        assert toks[idx] == ')'
        idx += 1  # skip ')'

    # intersect/union/except clause
    for op in SQL_OPS:  # initialize IUE
        sql[op] = None
    if idx < len_ and toks[idx] in SQL_OPS:
        sql_op = toks[idx]
        idx += 1
        idx, IUE_sql = parse_sql(toks, idx, tables_with_alias, schema)
        sql[sql_op] = IUE_sql
    return idx, sql


def remove_condition_values(nsql):
    if nsql['where']:
        for w in nsql['where']:
            if type(w) == list and type(w[3]) != list:
                w[3] = '"terminal"'
                if w[4] and type(w[4]) != list:
                    w[4] = '"terminal"'
    if nsql['limit']:
        nsql['limit'] = 1
    return nsql



def create_sql_from_natSQL(nsql, db_name, db, table_json, sq=None, remove_values=False, remove_groupby_from_natsql=False, args = Args()):
    find_table = table_json
    
    # Schema MODEL:
    schema = Schema_Star(get_schema(db, find_table))
    
    nsql = nsql.replace(" .*",".*")
    toks = tokenize_nSQL(nsql, None, False)

    tables_with_alias = get_tables_with_alias(schema.schema, toks)
    _, p_nsql = parse_sql(toks, 0, tables_with_alias, schema)

    if remove_values and sq:
        p_nsql = remove_condition_values(p_nsql)
        args.fill_value = True
    else:
        args.fill_value = False

    if remove_groupby_from_natsql:
        p_nsql['groupBy'] = []

    try:
        final_sql = inference_sql(p_nsql, find_table, args, sq=sq)
    except:
        return None, None, (None, None, find_table)
    
    return final_sql, p_nsql, (toks, tables_with_alias, find_table)

