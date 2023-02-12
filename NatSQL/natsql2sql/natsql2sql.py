from .utils import *
import copy
from .preprocess.stemmer import MyStemmer

SQL_TOP = 1 # top sql, based on 'except', 'intersect', 'union'
SQL_SUB = 2 # sub sql, 
SQL_SUBSUB = 3 # sub-sub sql, based on 'sub'
# WHERE_OPS = ('not', 'between', '=', '>', '<', '>=', '<=', '!=', 'in', 'like', 'is', 'exists', 'not in', 'not like', 'not between', 'join')

#NatSQL V1.1:
#add group by
#support where + IUE
#support move order by to subquery automatically
#extend join on: table_1.*=table_2.* -> join these tables. If table_1 == table_2, only join one more table.

#NatSQL V1.2:
#add un-foreign key search
#add @.@ column search for subquery
#add group by control for exact match

#NatSQL V1.3:
#separate the Join on
#add new key-word join
#improve group by accuracy for Spider

#NatSQL V1.4:
#implement the NatSQL extension
#implement question content analysis for groupBy and intersect
#implement a better fk analysis for IUE

#NatSQL V1.5:
#Further simplify the IUE SQL

#NatSQL V1.6:
#Optimize for table net

def natsql_version():
    return "1.6.1"

class Args():
    def __init__(self):
        self.in_sub_sql = True  # if True: whenever see "in col",it will be transferred to "in (select col)",else: the left and right col are foreign key relationship will be tranferred to "join on"
        self.eq_sub_sql = False # if True: whenever see "= col", it will be transferred to "= (select col)", else: only left and right col are not foreign key relationship will be tranferred
        self.keep_top_order_by = True # We previous plan to move "order by" to where condition. But now it is not used. So please keep it to be True.
        self.orderby_to_subquery = True # Move the order by to the subquery when there is subquery and this order by should be in the subquery.
        self.group_for_exact_match = True # For get a higher score in exact match in spider benchmark
        self.not_infer_group = True
        self.use_from_info = False
        self.print = False
        self.fill_value = True
        self.join2subquery = False
        self.iue2subquery = True
        self.groupby2subquery = False

class table_With_FK:
    def __init__(self, table, keys):
        self.table = table
        self.keys = keys

    def try_get_join_table(self, table_list, column):
        table_ = column[self.keys[0]][1].split('.')[0]
        if (table_.lower() in table_list and self.table.lower() not in table_list):
            return self.table
        if (table_.lower() not in table_list and self.table.lower() in table_list):
            return table_
        return None

    def return_join_on(self, table, column):
        sql = " join " + table
        sql += " on " + column[self.keys[0]][1] + " = " + column[self.keys[1]][1]
        return sql

globe_join_on_label_count = 0


def reversed_link_back_col(col_id, table_json):
    for lb in range(col_id, len(table_json['link_back'])):
        if table_json['link_back'][lb][1] == col_id:
            return table_json['link_back'][lb][0]
    return 0

def natsql_idx_to_table2_idx(col_id, table_json):
    return table_json['link_back'][col_id][1]

def check_relation(foreign_keys,primary_keys,column_names,left_column,right_column,left_tables=None):
    try:
        (k_1,k_2) = -1,-1
        if left_tables and not left_column: # means the left_column is @.@, so input set it to be None
            left_col = None
        else:
            left_col = val_unit_back(left_column).lower()
        right_col = col_unit_back(right_column).lower()
        for  idx, cn in enumerate(column_names):
            if left_col == cn[1].lower():
                k_1 = idx
            if right_col == cn[1].lower():
                k_2 = idx
        if right_col == "*" and k_2 == -1 and k_1 >= 0:
            right_col = right_column[1].split(".")[0].lower()
            for fk in foreign_keys:
                if fk[0] == k_1 and column_names[fk[1]][1].split('.')[0].lower() == right_col:
                    k_2 = fk[1]
                elif fk[1] == k_1 and column_names[fk[0]][1].split('.')[0].lower() == right_col:
                    k_2 = fk[0]
        elif not left_col and left_tables and k_1 == -1 and k_2 >= 0:
            for fk in foreign_keys:
                if fk[0] == k_2 and column_names[fk[1]][1].split('.')[0].lower() in left_tables:
                    k_1 = fk[1]
                elif fk[1] == k_2 and column_names[fk[0]][1].split('.')[0].lower() in left_tables:
                    k_1 = fk[0]
        elif right_col == "*" and left_col == "*" and k_2 == -1 and k_1 == -1:
            right_col = right_column[1].split(".")[0].lower()
            left_col = left_column[1][1].split(".")[0].lower()
            for fk in foreign_keys:
                if column_names[fk[0]][1].split('.')[0].lower() == right_col and column_names[fk[1]][1].split('.')[0].lower() == left_col:
                    return fk
                if column_names[fk[0]][1].split('.')[0].lower() == left_col and column_names[fk[1]][1].split('.')[0].lower() == right_col:
                    return fk
        if [k_1, k_2] in foreign_keys or [k_2, k_1] in foreign_keys:
            return [k_1, k_2]
        elif k_1 >= 0 and k_2 >= 0 and (k_1 in primary_keys or k_2 in primary_keys) and column_names[k_1][0] != column_names[k_2][0] and left_column[0]==0 and left_column[1][0] == 0 and right_column[0] == 0:
            return [k_1, k_2]
    except:
        pass
    return None


def condition_str(column):
    if WHERE_OPS[column[1]] == 'between':
        return  val_unit_back(column[2]) + " between " + str(column[3]) + ' and ' + str(column[4]) + " "
    elif type(column[3]) == list:
        return  val_unit_back(column[2]) + " " + WHERE_OPS[column[1]] + " " + column[3][1] + " "
    else:
        return  val_unit_back(column[2]) + " " + WHERE_OPS[column[1]] + " " + str(column[3]) + " "



def get_where_column(sql_dict, table_list, start_index, sql_type, table_json, args):
    AND_OR_TYPE = 1
    SUB_SQL_TYPE = 2
    SUB_SUB_SQL_TYPE = 3
    TOP_SQL_TYPE = AND_OR_TYPE

    break_idx = -1
    next_type = None
    last_column_type = None
    see_sub_sql = False

    left_col_list = []
    next_table_list = []
    sql_str = " "
    having = " "
    order_by = ''

    for idx, column in enumerate(sql_dict['where']):
        if idx < start_index:
            continue

        if isinstance(column,str) and column.lower() in SPECIAL_COND_OPS: # 'except', 'intersect', 'union', 'sub'
            if column == 'sub':
                last_column_type = SUB_SQL_TYPE
                break_idx,next_type = (idx + 1, SQL_SUBSUB) if (idx > start_index and not next_type) or (sql_type == SQL_SUB and not next_type) else (break_idx,next_type)
                # next_type = SQL_SUBSUB if idx > start_index and not next_type else next_type
                if sql_type == SQL_SUBSUB:
                    break
            else:
                break_idx,next_type = (idx, SQL_TOP) if (idx > start_index or break_idx==-1) and not next_type else (break_idx,next_type)
                break
        elif isinstance(column,str) and column.lower() in AND_OR_OPS:    # 'and', 'or'
            last_column_type = AND_OR_TYPE
            if sql_str.endswith('and '):
                sql_str =  sql_str[:-4]
            elif sql_str.endswith('or '):
                sql_str =  sql_str.strip()[:-3]
            if sql_str != " ":
                sql_str += column + " "
            continue
        else:
            assert not column[0]
            if type(column[3]) == list:
                if column[2][1][1] == "@.@":
                    fk = check_relation(table_json['foreign_keys'],table_json['primary_keys'],table_json['table_column_names_original'],None,column[3],table_list)
                else:
                    fk = check_relation(table_json['foreign_keys'],table_json['primary_keys'],table_json['table_column_names_original'],column[2],column[3])
                (table_left, column_left) = (c.lower() for c in column[2][1][1].split('.'))
                (table_right, column_right) = (c.lower() for c in column[3][1].split('.'))
                if (break_idx != -1 or next_type) and column[1] == 15:
                    continue
                elif column[1] == 15 and table_right.lower() not in table_list: #jion:
                    table_list.append(table_right.lower())
                elif column[1] == 15 and table_right.lower() in table_list: #jion:
                    continue
                elif ((not args.in_sub_sql and column[1] == 8) or (not args.eq_sub_sql and column[1] == 2 and table_left != "@")) and fk  :
                    if break_idx != -1 or next_type:
                        continue
                    if table_left.lower() not in table_list and table_left != "@": #V1.2
                        table_list.append(table_left.lower())
                    table = table_right # column[3][1].split('.')[0]
                    twfk = table_With_FK(table,fk)
                    table_list.append(twfk)
                elif table_left == table_right and column_left != column_right and column[1] >= 2 and column[1] <= 7 and not fk and (not (column[2][0] or column[2][1][0] or column[3][0])):
                    # different column in the same table
                    if break_idx != -1 or next_type or see_sub_sql:
                        continue
                    sql_str += condition_str(column) # for columnA != columnB in the same table
                elif column[2][0] or column[2][1][0] and column[3][0] in [1,2]:
                    # condition to order by
                    if break_idx != -1 or next_type or see_sub_sql:
                        continue
                    order_by = " ORDER BY " + val_unit_back(column[2]) + (" DESC LIMIT 1 " if column[3][0] == 1 else " ASC LIMIT 1 ")
                elif "*" in column_right and column[1] == 2 and column[3][0] == 0: #V1.2
                    # ? = * to join on
                    if break_idx != -1 or next_type or see_sub_sql:
                        continue
                    if table_left.lower() not in table_list and table_left != "@": #V1.2
                        table_list.append(table_left.lower())
                    if fk:
                        twfk = table_With_FK(table_right,fk)
                        table_list.append(twfk)
                    elif table_right.lower() not in table_list:
                        table_list.append(table_right.lower())
                else:
                    order_by_ = maybe_order_by(sql_dict, table_list, idx, start_index,  next_type,sql_type, table_json, args)
                    order_by += order_by_
                    if not order_by_:
                        break_idx,next_type = (idx,SQL_SUB) if (idx > start_index and not next_type) or (sql_type == SQL_TOP and not next_type) else (break_idx,next_type)
                        if last_column_type and sql_type > last_column_type: 
                            # (and + list) sql_type = 1 = new sub sql, it end 2 and 3(SQL_SUB, SQL_SUBSUB); sub + list = 2 = new subsub sql, it end 3(SQL_SUBSUB)
                            break
                        if last_column_type and sql_type != last_column_type: # it will true, when last is sub, and sql_type is 
                            # Filter by last if: 1.SQL_SUB meet 'and list sub sql model' 2. SQL_SUBSUB meet 'and list sub sql model' 3. SQL_SUBSUB meet 'sub list subsub sql model'
                            # It should True here when: 1. SQL_TOP meet 'sub list subsub sql model'
                            continue
                        # It should run here when: 1. SQL_TOP meet 'and list sub sql model'; 2. SQL_SUB meet 'sub list sub sql model'
                        see_sub_sql = True
                        if column[2][1][1] == "@.@": #V1.2
                            column = infer_at_col(column,table_list,table_json)
                            table_left = column[2][1][1].split(".")[0].lower()
                            if ".*" in column[2][1][1]:
                                column[2][1][1] = '*'
                        elif ".*" in column[3][1] and fk:
                            column[3][1] = table_json['table_column_names_original'][fk[1]][1] if column[2][1][1].lower() == table_json['table_column_names_original'][fk[0]][1].lower() else table_json['table_column_names_original'][fk[0]][1]
                        if column[2][0] or column[2][1][0]:
                            having += val_unit_back(column[2]).lower() + " " + WHERE_OPS[column[1]] + " (@@@) "
                        else:
                            if sql_str.strip() and not sql_str.strip().endswith("and") and not sql_str.strip().endswith("or"):
                                sql_str += " and "
                            sql_str += column[2][1][1].lower() + " " + WHERE_OPS[column[1]] + " (@@@) "
                table = table_left.lower()
                if not column[2][1][0] and column[2][1][1] != "@.@":
                    left_col_list.append(column[2][1][1].lower())
            elif not see_sub_sql:
                if not column[2][1][0]:
                    left_col_list.append(column[2][1][1].lower())
                table = column[2][1][1].split('.')[0].lower()
                if column[2][1][1] == "@.@" and idx >= 2: #V1.2
                    column[2][1][1] = sql_dict['where'][idx-2][2][1][1]
                if column[2][0] or column[2][1][0]:  # having
                    and_or = sql_dict['where'][idx - 1] + " " if idx>0 and type(sql_dict['where'][idx - 1])==str else None
                    if and_or and having != " ":
                        having += and_or
                    having += condition_str(column)
                    if and_or and sql_str.endswith(and_or):
                        sql_str = sql_str[:-len(and_or)]
                    pass
                else:                                # where
                    sql_str += condition_str(column)
                    

            if table and table.lower() not in table_list and table != "@": #V1.2
                table_list.append(table.lower())

    sql_str = sql_str if sql_str == " " else " where " + sql_str
    while True:
        if sql_str.strip().endswith('and'):
            sql_str = " " + sql_str.strip()[:-3] + " "
        elif sql_str.strip().endswith('or'):
            sql_str = " " + sql_str.strip()[:-2] + " "
        else:
            break

    having = having if having == " " else " having " + having 

    return break_idx, table_list, next_type, sql_str, having, order_by, next_table_list, left_col_list



def get_table_network(table_json, table_list, join_on_label, re_single=True, sq=None, sql_dict=None, group_list=None):

    def create_foreign_key(table_json, table_idx_list, restrict = True):
        """
        len(table_idx_list) must be two!!!
        """
        def restrict_check(table_json,pkey):
            if table_json['column_names'][pkey[0][0]][1] == "id" or table_json['column_names'][pkey[1][0]][1] == "id":
                return None
            if table_json['column_types'][pkey[0][0]] == table_json['column_types'][pkey[1][0]]:
                ws = table_json['column_names'][pkey[0][0]][1].split(" ")
                for w in ws:
                    if w in table_json['column_names'][pkey[1][0]][1]:
                        return [[[pkey[0][0],pkey[1][0]]],[pkey[0][1],pkey[1][1]]]
            return None


        def super_key_create_2_to_3(table_json,pkey):
            # try to find a new table as bridge to connect this two table
            for ii in range(2):
                pk_l = pkey[0][0] if ii == 0 else pkey[1][0]
                pk_r = pkey[1][0] if ii == 0 else pkey[0][0]
                table_l = table_json['column_names'][pk_l][0]
                table_r = table_json['column_names'][pk_r][0]
                for i,fk in enumerate(table_json['foreign_keys']):
                    if pk_l in fk:
                        other_fk = fk[1] if pk_l == fk[0] else fk[0]
                        # other_tble = table_json['column_names'][other_fk][0]
                        for j,fk2 in enumerate(table_json['foreign_keys']):
                            if i == j:
                                continue
                            if other_fk in fk2: # jump here:
                                jump_fk = fk2[1] if other_fk == fk2[0] else fk2[0]
                                jump_tble = table_json['column_names'][jump_fk][0]
                                for k,fk3 in enumerate(table_json['foreign_keys']):
                                    if k in (i,j):
                                        continue
                                    if pk_r in fk3:
                                        jump_fk2 = fk3[1] if pk_r == fk3[0] else fk3[0]
                                        if table_json['column_names'][jump_fk2][0] ==  jump_tble:
                                            return [[[pk_l,jump_fk],fk3],[table_l,jump_tble,table_r]]
            return None
                                

        pkey = [] # try to directly use the primary key to join
        for k in table_json['primary_keys']:
            if table_json['column_names'][k][0] in table_idx_list:
                pkey.append([k,table_json['column_names'][k][0]])
        potential_fk = []
        for pk in pkey: # same name with primary key to become JOIN ON
            other_table = table_idx_list[1] if pk[1] == table_idx_list[0] else table_idx_list[0]
            for i,o_col,col in zip(range(len(table_json['column_names_original'])),table_json['column_names_original'],table_json['column_names']):
                if col[0] == other_table and (table_json['link_back'][pk[0]][1] in table_json['same_col_idxs'][table_json['link_back'][i][1]] or table_json['link_back'][i][1] in table_json['same_col_idxs'][table_json['link_back'][pk[0]][1]]):
                    potential_fk.append([[[pk[0],i]],[pk[1],other_table]])
        if potential_fk:
            return potential_fk[0]
        else:
            if len(pkey) == 2:
                if restrict:
                    r_ = restrict_check(table_json,pkey)
                    if r_:
                        return r_
                else:
                    tale_net = super_key_create_2_to_3(table_json,pkey)
                    if tale_net:
                        return tale_net
                    return [[[pkey[0][0],pkey[1][0]]],[pkey[0][1],pkey[1][1]]]
            # try to directly use the same column name key to join
            for i, c_n,c_on in zip(range(len(table_json['column_names'])), table_json['column_names'],table_json['column_names_original']):
                if c_n[0] == table_idx_list[0]:
                    for j,c_n2,c_on2 in zip(range(len(table_json['column_names'])), table_json['column_names'],table_json['column_names_original']):
                        if c_n2[0] == table_idx_list[1]:
                            if table_json['column_types_checked'][j] == table_json['column_types_checked'][i] and ((c_n[1] ==  c_n2[1] and c_n[1] not in ["id","name","ids","*"]) or (c_on[1] ==  c_on2[1] and c_on[1] not in ["id","name","ids","*"])):
                                return [[[i,j]],table_idx_list]

            # try to directly use the foreign key to join
            r_ = None 
            pkey = []
            tables = []
            for k in table_json['foreign_keys']:
                if table_json['column_names'][k[0]][0] in table_idx_list and table_json['column_names'][k[0]][0] not in tables:
                    pkey.append([k[0],table_json['column_names'][k[0]][0]])
                    tables.append(table_json['column_names'][k[0]][0])
                if table_json['column_names'][k[1]][0] in table_idx_list and table_json['column_names'][k[1]][0] not in tables:
                    pkey.append([k[1],table_json['column_names'][k[1]][0]])
                    tables.append(table_json['column_names'][k[1]][0])
            if len(pkey) == 2:
                if restrict:
                    r_ = restrict_check(table_json,pkey)
                else:        
                    return [[[pkey[0][0],pkey[1][0]]],[pkey[0][1],pkey[1][1]]]
            
            if not r_:
                pkey = []
                for k in table_json['original_primary_keys']:
                    if table_json['column_names'][k][0] in table_idx_list:
                        pkey.append([k,table_json['column_names'][k][0]])
                if len(pkey) == 2:
                    if restrict:
                        r_ = restrict_check(table_json,pkey)
                        if r_:
                            return r_
                    else:
                        tale_net = super_key_create_2_to_3(table_json,pkey)
                        if tale_net:
                            return tale_net
                        return [[[pkey[0][0],pkey[1][0]]],[pkey[0][1],pkey[1][1]]]
        return r_
            
    def get_fk_network(table_json, table_list, return_num=1):
        table_index_list = [] # change table name list to a number list
        table_fk_list = []

        for t in table_list:
            if isinstance(t,table_With_FK):
                table_fk_list.append(t)
            else:
                if t.lower() != '@': #V1.2
                    table_index_list.append([n.lower() for n in table_json['table_names_original']].index(t.lower()))#(table_json['table_names_original'].index(t))
        
        from_table_net = []
        for idx, network in enumerate(table_json['network']):
            if len(network[1]) == len(table_index_list) or (len(network[1]) > len(table_index_list) and len(table_index_list)>1):
                success = True
                for t in table_index_list:
                    if t not in network[1]:
                        success = False
                        break
                if success and network[1][0] in table_index_list and network[1][-1] in table_index_list:
                    from_table_net.append(copy.deepcopy(network))
        if not from_table_net:
            for idx, network in enumerate(table_json['network']):
                if len(network[1]) == len(table_index_list) or (len(network[1]) > len(table_index_list) and len(table_index_list)>1):
                    success = True
                    for t in table_index_list:
                        if t not in network[1]:
                            success = False
                            break
                    if success:
                        from_table_net.append(copy.deepcopy(network))
        # re-order
        if from_table_net:
            from_table_net = sorted(from_table_net,key=lambda x:len(x[1]))

        if from_table_net and table_fk_list:
            for net in from_table_net:
                for tf in table_fk_list:
                    if (table_json['column_names'][tf.keys[0]][0] in net[1] and table_json['column_names'][tf.keys[1]][0] not in net[1]) or (table_json['column_names'][tf.keys[1]][0] in net[1] and table_json['column_names'][tf.keys[0]][0] not in net[1]):
                        net[0].append(tf.keys)
                        net[1].append([n.lower() for n in table_json['table_names_original']].index(tf.table.lower()))
        if return_num and from_table_net:
            return from_table_net[0], table_fk_list, table_index_list 
        else:
            return from_table_net, table_fk_list, table_index_list 
    

    def analyze_table_net_with_sq(sq,from_table_net,table_index_list,table_json,sql_dict):
        def table_not_in_db_match(sq, t_id, sub_idx, table_json):
            for dbms in sq.db_match[sub_idx]:
                for dbm in dbms:
                    col_id = reversed_link_back_col(dbm, table_json)
                    if table_json['column_names'][col_id][0] == t_id and "name" in table_json['column_names'][col_id][1]:
                        return False
                    if table_json['table_names'][t_id].split(" | ")[0] in table_json['column_names'][col_id][1]:
                        return False
            return True
        def there_is_not_close_column(sq, t_id, sub_idx, sub_idx2, table_json):
            if sub_idx2+1 < len(sq.col_match[sub_idx]) and sq.col_match[sub_idx][sub_idx2+1]:
                return False
            if sub_idx2-1 > 0 and sq.col_match[sub_idx][sub_idx2-2] and sq.sub_sequence_toks[sub_idx][sub_idx2-1] == "of":
                for col in sq.col_match[sub_idx][sub_idx2-2]:
                    col_id = reversed_link_back_col(col[0], table_json)
                    if table_json['column_names'][col_id][0] == t_id and "name" in table_json['column_names'][col_id][1]:
                        return False
                    if table_json['table_names'][t_id].split(" | ")[0] in table_json['column_names'][col_id][1]:
                        return False
            return True
        def table_in_original(net_tables,t_matchs,except_t,with_weight=False):
            for t in t_matchs:
                if with_weight:
                    if t != except_t and t in net_tables:
                        return False
                else:
                    if t[0] != except_t and t[0] in net_tables:
                        return False
            return True
        def the_word_is_not_a_column(sq, t_id, sub_idx, sub_idx2, table_json):
            if sq.col_match[sub_idx][sub_idx2]:
                if hasattr(sq,"table_match_weight"):
                    for t,tw in zip(sq.table_match[sub_idx][sub_idx2],sq.table_match_weight[sub_idx][sub_idx2]):
                        if t == t_id and tw != 1:
                            return False
                else:
                    for t in sq.table_match[sub_idx][sub_idx2]:
                        if t[0] == t_id and t[1] != 1:
                            return False
                for t in sq.col_match[sub_idx][sub_idx2]:
                    if t[1] == 1:
                        return False
            return True
        def the_table_is_not_a_column_in_sql_dict(sql_dict, table_json, t_id):
            t_word = table_json['table_names'][t_id].lower()
            for select in sql_dict['select'][1]:
                if select[1][1][1] in table_json['tc_fast']:
                    c_id = table_json['tc_fast'].index(select[1][1][1])
                    if t_word + " " in table_json['column_names'][c_id][1].lower() or " " + t_word  in table_json['column_names'][c_id][1].lower():
                        return False
            for where in sql_dict['where']:
                if type(where) == list:
                    if where[2][1][1] in table_json['tc_fast']:
                        c_id = table_json['tc_fast'].index(where[2][1][1])
                        if t_word + " " in table_json['column_names'][c_id][1].lower() or " " + t_word  in table_json['column_names'][c_id][1].lower():
                            return False
                    if type(where[3]) == list and where[3][1] in table_json['tc_fast']:
                        c_id = table_json['tc_fast'].index(where[3][1])
                        if t_word + " " in table_json['column_names'][c_id][1].lower() or " " + t_word  in table_json['column_names'][c_id][1].lower():
                            return False
            if sql_dict['orderBy']:
                if sql_dict['orderBy'][1][0][1][1] in table_json['tc_fast']:
                    c_id = table_json['tc_fast'].index(sql_dict['orderBy'][1][0][1][1])
                    if t_word + " " in table_json['column_names'][c_id][1].lower() or " " + t_word  in table_json['column_names'][c_id][1].lower():
                        return False
            return True

        def use_the_net_col_appear_in_question(table_net,idx1,idx2,sq,table_json):
            n1 = set([c for cols in table_net[idx1][0] for c in cols])
            n2 = set([c for cols in table_net[idx2][0] for c in cols])
            cols1 = n1.difference(n2)
            cols2 = n2.difference(n1)
            cols1_str = (" ".join([table_json["column_names"][col][1] for col in cols1])).split(" ")
            cols2_str = (" ".join([table_json["column_names"][col][1] for col in cols2])).split(" ")
            cols1 = set(cols1_str).difference(set(cols2_str))
            cols2 = set(cols2_str).difference(set(cols1_str))
            count1 = 0
            count2 = 0
            for w in cols1:
                if w in " " + sq.question_or + " " or w in " " + sq.question_lemma + " ":
                    count1 += 1
            for w in cols2:
                if w in " " + sq.question_or + " " or w in " " + sq.question_lemma + " ":
                    count2 += 1
            if count1 > count2:
                return idx2
            elif count1 < count2:
                return idx1
            if "source" in cols1 or "source" in cols2:
                if "source" in cols2 and "from" in " " + sq.question_or + " ":
                    return idx1
            return idx2


        if sq and len(from_table_net)>1:
            sort_from_table_net = copy.deepcopy(from_table_net)
            sort_table_index_list = copy.deepcopy(table_index_list)
            min_table = len(from_table_net[0][1])
            sort_from_table_net[0][1].sort()
            del_idxs = set()
            for i in range(len(from_table_net)-1,0,-1):
                if len(from_table_net[i][1]) > min_table + 1:
                    del_idxs.add(i)
                else:
                    sort_from_table_net[i][1].sort()
                    if sort_from_table_net[i][1] == sort_from_table_net[0][1]:
                        del_idxs.add(use_the_net_col_appear_in_question(sort_from_table_net,i,0,sq,table_json))
            if len(from_table_net) >= 3:
                for j in range(1, len(from_table_net)-1):
                    sort_from_table_net[j][1].sort()
                    for i in range(len(from_table_net)-1,0,-1):
                        if i != j:
                            sort_from_table_net[i][1].sort()
                            if sort_from_table_net[i][1] == sort_from_table_net[j][1]:
                                del_idxs.add(use_the_net_col_appear_in_question(sort_from_table_net,i,j,sq,table_json))
            if len(list(del_idxs)) == len(from_table_net):
                return [from_table_net[0]]
            if del_idxs:
                del_idxs = list(del_idxs)
                del_idxs.sort(reverse=True)
                for idx in del_idxs:
                    del from_table_net[idx]
                    del sort_from_table_net[idx]

            if len(from_table_net) == 1:
                return from_table_net

            # extract difference
            sort_table_index_list.sort()
            differen_list = []
            for i, net_s in enumerate(sort_from_table_net):
                diff_now = []
                if net_s[1] == sort_table_index_list:
                    differen_list.append(None)
                else:
                    for j, net_s2 in enumerate(sort_from_table_net):
                        if diff_now and len(net_s2[1]) != min_table:
                            break
                        if i != j:
                            tmp = set(net_s[1]).difference(set(net_s2[1]))
                            diff_now.extend(list(tmp))
                    differen_list.append(set(diff_now))


            # calc difference weight
            differen_weight = []
            for diff, net in zip(differen_list,from_table_net):
                if diff:
                    total = 0
                    for d in diff:
                        max_d = 0
                        if hasattr(sq,"table_match_weight"):
                            for (i,tm),tws in zip(enumerate(sq.table_match),sq.table_match_weight):
                                for (j,ts),twl in zip(enumerate(tm),tws):
                                    for t,tw in zip(ts,twl):
                                        if t == d and tw > max_d and the_word_is_not_a_column(sq, d, i, j, table_json) and table_in_original(net[1],ts,d,True) and table_not_in_db_match(sq,d,i,table_json) and there_is_not_close_column(sq, d, i, j, table_json) and 'NOT' not in sq.pattern_tok[i] and the_table_is_not_a_column_in_sql_dict(sql_dict, table_json, d):
                                            max_d = tw
                        else:
                            for i,tm in enumerate(sq.table_match):
                                for j,ts in enumerate(tm):
                                        for t in ts:
                                            if t[0] == d and t[1] > max_d and the_word_is_not_a_column(sq, d, i, j, table_json) and table_in_original(net[1],ts,d) and table_not_in_db_match(sq,d,i,table_json) and there_is_not_close_column(sq, d, i, j, table_json) and 'NOT' not in sq.pattern_tok[i] and the_table_is_not_a_column_in_sql_dict(sql_dict, table_json, d):
                                                max_d = t[1]
                        if max_d == 0:
                            total -= 0.05
                        total += max_d
                    differen_weight.append(total/len(diff))
                else:
                    differen_weight.append(0.9)
            import operator
            max_index, max_number = max(enumerate(differen_weight), key=operator.itemgetter(1))
            return [from_table_net[max_index]]
        return from_table_net


    ##########################################################

    from_table_net, table_fk_list, table_index_list = get_fk_network(table_json, table_list, 0)

    from_table_net = analyze_table_net_with_sq(sq, from_table_net, table_index_list, table_json, sql_dict)
        
    if not from_table_net and len(table_index_list) == 2:
        if re_single:
            return create_foreign_key(table_json, table_index_list,False),table_fk_list
        else:
            return [create_foreign_key(table_json, table_index_list,False)],table_fk_list
    elif not from_table_net and len(table_index_list) == 3:
        from_table_net1, table_fk_list1, table_index_list1 = get_fk_network(table_json, [table_list[0],table_list[1]])
        from_table_net2, table_fk_list2, table_index_list2 = get_fk_network(table_json, [table_list[0],table_list[2]])
        from_table_net3, table_fk_list3, table_index_list3 = get_fk_network(table_json, [table_list[1],table_list[2]])
        net1 = None
        net2 = None
        net3 = None
        if not from_table_net1 and not from_table_net2 and from_table_net3:
            net1 = create_foreign_key(table_json, [table_index_list[0],table_index_list[1]])
            net2 = create_foreign_key(table_json, [table_index_list[0],table_index_list[2]])
            net3 = copy.deepcopy(from_table_net3) 
        elif not from_table_net1 and  from_table_net2 and not from_table_net3:
            net1 = create_foreign_key(table_json, [table_index_list[0],table_index_list[1]])
            net2 = create_foreign_key(table_json, [table_index_list[2],table_index_list[1]])
            net3 = copy.deepcopy(from_table_net2) 
        elif from_table_net1 and not from_table_net2 and not from_table_net3:
            net1 = create_foreign_key(table_json, [table_index_list[1],table_index_list[2]])
            net2 = create_foreign_key(table_json, [table_index_list[0],table_index_list[2]])
            net3 = copy.deepcopy(from_table_net1) 
        elif from_table_net1 and from_table_net2 and not from_table_net3:
            net1 = copy.deepcopy(from_table_net1) 
            net3 = copy.deepcopy(from_table_net2) 
        elif from_table_net1 and not from_table_net2 and from_table_net3:
            net1 = copy.deepcopy(from_table_net1) 
            net3 = copy.deepcopy(from_table_net3) 
        elif not from_table_net1 and from_table_net2 and from_table_net3:
            net1 = copy.deepcopy(from_table_net3) 
            net3 = copy.deepcopy(from_table_net2) 
        else:
            return from_table_net, table_fk_list

        if net1 and not net2:
            # combine the net1 
            net3[0].append(net1[0][0])  
            net3[1].append(net1[1][0] if net1[1][0] not in net3[1] else net1[1][1])
        elif not net1 and net2:
            net3[0].append(net2[0][0])  
            net3[1].append(net2[1][0] if net2[1][0] not in net3[1] else net2[1][1])
        elif net1 and net2:
            if net1[0][0][0] in table_json["primary_keys"] or net1[0][0][1] in table_json["primary_keys"]:
                net2 = net1
            net3[0].append(net2[0][0])  
            net3[1].append(net2[1][0] if net2[1][0] not in net3[1] else net2[1][1])
        
        if len(net3[1]) == 3:
            if re_single:
                return net3,table_fk_list
            else:
                return [net3],table_fk_list

    idx = 0
    if from_table_net and group_list:
        table_list = [table_json["column_names"][table_json["tc_fast"].index(col)][0] for col in group_list]
        table_list = list(set(table_list))
        if len(table_list) >= 1:
            for i, net in enumerate(from_table_net):
                join_cols = [c for cols in net[0] for c in cols if c in table_json['primary_keys'] and table_json['column_names'][c][0] in table_list ]
                if join_cols:
                    idx = i 
                    break
    if from_table_net and re_single:
        if join_on_label:
            global globe_join_on_label_count
            idx = join_on_label[globe_join_on_label_count]
            globe_join_on_label_count += 1
        else:
            idx = 0
        return from_table_net[idx], table_fk_list
    return from_table_net, table_fk_list



def create_from_table(from_table_net,table_names, column, table_fk_list, first_table=None, table_list=None):
    if not from_table_net and Args().print:
        print("Can not find correct table network")
        pass
    elif not from_table_net:
        if table_list:
            return " from " + " join ".join(table_list)
        return " from " + table_names[0] + " "
    assert from_table_net, "Can not find correct table network"
    table_use = []
    table_fk_remain = []
    sql = None

    # for spider sub query bug. Actually this is useless
    if first_table and len(from_table_net[0]) == 1 and len(from_table_net[1]) == 2:
        from_table_net = copy.deepcopy(from_table_net)
        if table_names[from_table_net[1][0]].lower() != first_table.lower():
            from_table_net[1] = [from_table_net[1][1],from_table_net[1][0]]
        if column[from_table_net[0][0][0]][0] != from_table_net[1][0]:
            from_table_net[0][0] = [from_table_net[0][0][1],from_table_net[0][0][0]]



    if not from_table_net[0]: # only one table
        sql = " from " + table_names[from_table_net[1][0]]
        table_use.append(table_names[from_table_net[1][0]].lower())

    table_fk_remain = copy.deepcopy(table_fk_list) 

    for fk in from_table_net[0]:
        if not sql:
            sql = " from " + table_names[column[fk[0]][0]]
            table_use.append(table_names[column[fk[0]][0]].lower())
        
        table_fk_list = []
        for tfk in table_fk_remain: # it is impossible to replace one network when start table is not table_names[from_table_net[1][0]] in this code. it can be update later.
            join_table = tfk.try_get_join_table(table_use,column)
            if join_table:
                sql += tfk.return_join_on(join_table, column)
                table_use.append(join_table.lower())
            else:
                table_fk_list.append(tfk)
        table_fk_remain = copy.deepcopy(table_fk_list)

        if table_names[column[fk[1]][0]].lower() not in table_use:
            sql += " join " + table_names[column[fk[1]][0]]
            sql += " on " + column[fk[0]][1] + " = " + column[fk[1]][1]
            table_use.append(table_names[column[fk[1]][0]].lower())
        elif table_names[column[fk[0]][0]].lower() not in table_use:
            sql += " join " + table_names[column[fk[0]][0]]
            sql += " on " + column[fk[1]][1] + " = " + column[fk[0]][1]
            table_use.append(table_names[column[fk[0]][0]].lower())

    for tfk in table_fk_remain:
        join_table = tfk.try_get_join_table(table_use,column)
        if join_table:
            sql += tfk.return_join_on(join_table, column)
            table_use.append(join_table.lower())
    
    return sql


def create_order_by(order_dict,limit):
    orderby = ""
    table_list = []
    agg_in_order_bool = False
    if order_dict:
        orderby = " order by "
        orderby += ",".join([val_unit_back(order) for order in order_dict[1]])
        orderby += " " + order_dict[0]

        agg_in_order = [val_unit_contain_agg(order) for order in order_dict[1]]
        agg_in_order_bool = True if True in agg_in_order else False

        for order in order_dict[1]:
            table = order[1][1].split('.')[0].lower()
            if table not in table_list:
                table_list.append(table)
    if limit:
        orderby += " limit " + str(limit) + " "
    return orderby,table_list, agg_in_order_bool


def maybe_order_by(sql_dict, table_list, idx, start_index, next_sql_type, sql_type_now, table_json, args):
    """
     order by and sub sql ->
     order by -> return order by
     and sub-sql and order-by 
     and sub-sql sub subsub-sql

    """

    if (idx == start_index and sql_type_now != SQL_TOP) or (next_sql_type == SQL_SUBSUB and idx > 0 and sql_dict['where'][idx-1] == 'sub'):
        return ''

    if (sql_dict['where'][idx][3][0] == 1 or sql_dict['where'][idx][3][0] == 2) and sql_dict['where'][idx][2][1][1] == sql_dict['where'][idx][3][1] and sql_dict['where'][idx][1] == 2: # sql_dict['where'][idx][1] == 2 means '='
        break_idx,next_type = (idx, SQL_SUB) if (idx > start_index and not next_sql_type) or (sql_type_now == SQL_TOP and not next_sql_type) else (idx,next_sql_type)
        break_idx,table_list,next_sql,sql_where,sql_having,orderby_sql_,next_table_list,_ = get_where_column(sql_dict, table_list, break_idx + 1, next_type, table_json, args)
        if (not next_sql and len(sql_dict['where']) == idx + 1) or (next_sql and break_idx <= idx + 2): # the end of the sql or sub sql.
            if next_sql_type: # make it pass for next if check in get_where_column
                return ' '
            
            if args.keep_top_order_by and sql_type_now == SQL_TOP:
                return ''

            order_by_sql = " ORDER BY " + sql_dict['where'][idx][2][1][1] + (" DESC LIMIT 1 " if sql_dict['where'][idx][3][0] == 1 else " ASC LIMIT 1 ")
            return order_by_sql
    return ''



def is_there_subquery(wheres):
    """
        V1.1
    """
    if not wheres:
        return False
    for where in wheres:
        if isinstance(where,list) and isinstance(where[3],list):
            if where[2][1][1] == "@.@" and "*" in where[3][1] and where[1] == 2:
                continue
            return True
    return False



def is_orderby_for_subquery(sql_dict):
    """
        V1.1
    """
    if not sql_dict['limit'] or not sql_dict['orderBy']:
        return False
    if is_there_subquery(sql_dict['where']):
        return True
    return False


def orderby_to_subquery(sql_dict,tb_list):
    """
        V1.1
    """
    orderby_sql = ""
    if is_orderby_for_subquery(sql_dict):
        orderby_sql,table_list,agg_in_order = create_order_by(sql_dict['orderBy'],sql_dict['limit'])
        for t in table_list:
            if t not in tb_list:
                tb_list.append(t)
    return orderby_sql,tb_list


def primary_keys(table_json,table_id,pk_only=False):
    for key in table_json['primary_keys']:
        if table_json['column_names'][key][0] == table_id:
            return key
    for key in table_json['original_primary_keys']:
        if table_json['column_names'][key][0] == table_id:
            return key
    if pk_only and table_json['primary_keys']:
        return -1
    for key,col in enumerate(table_json['column_names']):
        if table_json['column_names'][key][0] == table_id:
            col_n = table_json['column_names'][key][1].lower().strip()
            col_n = col_n.replace(table_json['table_names'][table_id].lower(),"").strip()
            col_n = col_n.replace(table_json['table_names_original'][table_id].lower(),"").strip()
            if col_n  in ["id","ids"]:
                return key
    return -1

def contain_bridge_table(table_list, bridge_tables):
    for b in bridge_tables:
        if b in table_list:
            return True
    return False


def infer_at_col(column,table_list,table_json):
    """V1.2
    infer the @.@ column value
    """
    def find_fk_cols(l_table,r_table,table_json,r_col=None):
        if r_col:
            for fk in table_json['foreign_keys']:
                if r_col == fk[0] and table_json['column_names'][fk[1]][0] == l_table:
                    return True,fk[1],r_col
                elif r_col == fk[1] and table_json['column_names'][fk[0]][0] == l_table:
                    return True,fk[0],r_col
        else:
            for fk in table_json['foreign_keys']:
                if table_json['column_names'][fk[0]][0] == r_table and table_json['column_names'][fk[1]][0] == l_table:
                    return True,fk[1],fk[0]
                elif table_json['column_names'][fk[1]][0] == r_table and table_json['column_names'][fk[0]][0] == l_table:
                    return True,fk[0],fk[1]
            for net in table_json['network']:
                if len(net[1]) == 3 and r_table in net[1] and l_table in net[1] and r_table not in table_json['bridge_table'] and l_table not in table_json['bridge_table'] and contain_bridge_table(net[1], table_json['bridge_table']):
                    for fk in net[0]:
                        if table_json['column_names'][fk[0]][0] == l_table and table_json['column_names'][fk[1]][0] in table_json['bridge_table']:
                            return True,fk[0],fk[1]
                        elif table_json['column_names'][fk[1]][0] == l_table and table_json['column_names'][fk[0]][0] in table_json['bridge_table']:
                            return True,fk[1],fk[0]
        return False,0,0

    def find_col(l_tables,r_table,r_col,table_json):
        # foreign key search:
        r_col_idx = 0
        if r_col != "*":
            for i,col in enumerate(table_json['column_names_original']):
                if col[0] == r_table and r_col.lower() == col[1].lower():
                    r_col_idx = i
                    break
        
        for lt in l_tables[1]:
            sucess,left,right = find_fk_cols(lt,r_table,table_json,r_col_idx)
            if sucess:
                return left,right

        # same name key search:
        if r_col != "*":
            for lt in l_tables[1]:
                for i,col_o,col in zip(range(len(table_json['column_names_original'])),table_json['column_names_original'],table_json['column_names']):
                    if col_o[0] == lt and (col_o[1].lower() == r_col or (col[1] == table_json['column_names'][r_col_idx][1] and col[1] not in ["name","names","id","ids"]) or col[1].lower() == (table_json['table_names'][table_json['column_names'][r_col_idx][0]] + " " + table_json['column_names'][r_col_idx][1]).lower() or (table_json['table_names'][col[0]] + " " +  col[1]).lower() == table_json['column_names'][r_col_idx][1].lower() ):
                        return i,r_col_idx
            if table_json['column_names'][r_col_idx][1].count(" ") == 2:
                # three match two to return
                col_r_names = table_json['column_names'][r_col_idx][1].split(" ")
                for lt in l_tables[1]:
                    for i,col in zip(range(len(table_json['column_names_original'])),table_json['column_names']):
                        if col[0] == lt and (col[1].count(" ") == 1):
                            c_ls = col[1].split(" ")
                            if c_ls[0] in col_r_names and c_ls[1] in col_r_names:
                                return i,r_col_idx
                    for i,col in zip(range(len(table_json['column_names_original'])),table_json['column_names']):
                        if col[0] == lt and (col[1].count(" ") == 0):
                            if col[1] in col_r_names and table_json["table_names"][lt] in col_r_names and col[1] != table_json["table_names"][lt]:
                                return i,r_col_idx
            return r_col_idx,r_col_idx
        else:
            result = []
            for j,rcol_o,rcol in zip(range(len(table_json['column_names_original'])),table_json['column_names_original'],table_json['column_names']):
                if rcol[0] != r_table or rcol_o[1] == '*':
                    continue
                for lt in l_tables[1]:
                    for i,col_o,col in zip(range(len(table_json['column_names_original'])),table_json['column_names_original'],table_json['column_names']):
                        if col_o[0] == lt and ((col_o[1].lower() == rcol_o[1].lower() and col_o[1].lower() not in ["name","names","id","ids"]) or (col[1].lower() == rcol[1].lower() and col[1].lower() not in ["name","names","id","ids"]) or ( (table_json['table_names'][col[0]] + " " + col[1]).lower() == rcol[1].lower() or (table_json['table_names'][rcol[0]] + " " + rcol[1]).lower() == col[1].lower() ) ):
                            result.append([i,j])
            if result:
                for r in result:
                    # if table_json['column_names_original'][r[0]][1].lower() not in ["name","id"] and table_json['column_names'][r[0]][1].lower() not in ["name","id"]:
                    if r[0] in table_json['primary_keys'] or r[1] in table_json['primary_keys']:
                        return r[0],r[1]
                for r in result:
                    if table_json['column_names_original'][r[0]][1].lower() not in ["name","id"] and table_json['column_names_original'][r[0]][1] == table_json['column_names_original'][r[1]][1]:
                        return r[0],r[1]
                return result[0][0],result[0][1]
        if r_table >= 0:
            # use the right table:
            pl = primary_keys(table_json,r_table)
            if pl >= 0:
                return pl,pl
            if len(l_tables[1]) >= 1:
                pl = primary_keys(table_json,l_tables[1][0])
            if pl >= 0:
                return pl,pl
        return 0,0
    
    col_right = column[3][1].split(".")
    table_right = col_right[0].lower()
    col_right = col_right[1].lower()
    table_right_idx = [n.lower() for n in table_json['table_names_original']].index(table_right)
    if table_right in table_list: # It will be the same column for both side
        if col_right == "*":
            primarykey = primary_keys(table_json,table_right_idx)
            if primarykey >= 0:
                column[3][1] = table_json['table_column_names_original'][primarykey][1]
        column[2][1][1] = column[3][1]
    else:
        if len(table_list) > 1:
            from_table_net,table_fk_list = get_table_network(table_json, table_list, None)
            if from_table_net and table_right_idx in from_table_net[1]:
                # There is same tables
                if col_right == "*":
                    primarykey = primary_keys(table_json,table_right_idx)
                    if primarykey >= 0:
                        column[3][1] = table_json['table_column_names_original'][primarykey][1]
                column[2][1][1] = column[3][1]
            elif from_table_net:
                # There isn't same tables, so look for the foreign key relations.
                col_left_idx,col_right_idx = find_col(from_table_net,table_right_idx,col_right,table_json)
                column[3][1] = table_json['table_column_names_original'][col_right_idx][1]
                column[2][1][1] = table_json['table_column_names_original'][col_left_idx][1]
        else:
            # There isn't same tables, so look for the foreign key relations.
            table_left_idx = [n.lower() for n in table_json['table_names_original']].index(table_list[0].lower())
            col_left_idx,col_right_idx = find_col([[],[table_left_idx]],table_right_idx,col_right,table_json)
            column[3][1] = table_json['table_column_names_original'][col_right_idx][1]
            column[2][1][1] = table_json['table_column_names_original'][col_left_idx][1]
    return column


def intersect_where_order(wheres,top_select_table_list):
    if ((len(wheres) == 3 and wheres[1] == "and") ) and wheres[0][1] == wheres[2][1] and wheres[0][1] == 7:
        if wheres[0][2][1][1].split(".")[0] not in top_select_table_list and wheres[2][2][1][1].split(".")[0] in top_select_table_list:
            (wheres[0],wheres[2]) = (wheres[2],wheres[0])
    if ((len(wheres) == 3 and wheres[1] == "or") ) and wheres[0][1] == wheres[2][1] and wheres[0][1] == 2:
        left = wheres[0][2][1][1].split(".")
        right = wheres[2][2][1][1].split(".")
        if left[0] not in top_select_table_list and right[0] in top_select_table_list and left[1] == right[1]:
            (wheres[0],wheres[2]) = (wheres[2],wheres[0])
    return wheres




def intersect_inference(wheres,sq):
    def replace_str_for_inference(wheres,where_idx_1,where_idx_2,sq,middle=0):
        if sq and wheres[where_idx_1][1] in [4,6,3,5] and wheres[where_idx_2][1] in [4,6,3,5] and isinstance(wheres[where_idx_1][3],str) and isinstance(wheres[where_idx_2][3],str):
            num_count = 0
            for toks in sq.pattern_tok:
                num_count += toks.count("NUM")
                num_count += toks.count("YEAR")
            if num_count == 2:
                for toks,sub_q in zip(sq.pattern_tok,sq.sub_sequence_toks):
                    for tok,q_tok in zip(toks,sub_q):
                        if tok in ["NUM","YEAR"]:
                            wheres[where_idx_1 if num_count==2 else where_idx_2][3] = str2num(q_tok)
                            num_count -= 1
        middle = int((where_idx_1 + where_idx_2)/2) if not middle else middle
        if wheres[where_idx_1][1] == 2 and wheres[where_idx_2][1] == 2 and not (wheres[where_idx_2][2][1][0]==3 and wheres[where_idx_1][2][1][0]==0):
            wheres[middle] = "intersect_"
        elif ".*" in wheres[where_idx_1][2][1][1] and ".*" in wheres[where_idx_2][2][1][1]:
            wheres[middle] = "intersect_"
        elif (wheres[where_idx_2][2][1][0] in [1,2,4,5,6]  and wheres[where_idx_1][1] not in [8,12,15]) or (wheres[where_idx_2][2][1][0] and wheres[where_idx_1][2][1][0]):
            wheres[middle] = "intersect_"
        elif wheres[where_idx_1][1] == 4 and wheres[where_idx_2][1] == 3 and isinstance(wheres[where_idx_1][3],str) and isinstance(wheres[where_idx_2][3],str) and wheres[where_idx_1][2][1] == wheres[where_idx_2][2][1]:
            wheres[middle] = "intersect_"
        elif wheres[where_idx_1][1] == 6 and wheres[where_idx_2][1] == 5 and isinstance(wheres[where_idx_1][3],str) and isinstance(wheres[where_idx_2][3],str) and wheres[where_idx_1][2][1] == wheres[where_idx_2][2][1]:
            wheres[middle] = "intersect_"
        elif wheres[where_idx_1][1] == 6 and wheres[where_idx_2][1] == 3 and isinstance(wheres[where_idx_1][3],str) and isinstance(wheres[where_idx_2][3],str) and wheres[where_idx_1][2][1] == wheres[where_idx_2][2][1]:
            wheres[middle] = "intersect_"
        elif wheres[where_idx_1][1] == 4 and wheres[where_idx_2][1] == 5 and isinstance(wheres[where_idx_1][3],str) and isinstance(wheres[where_idx_2][3],str) and wheres[where_idx_1][2][1] == wheres[where_idx_2][2][1]:
            wheres[middle] = "intersect_"
        #################################
        elif wheres[where_idx_1][1] == 3 and wheres[where_idx_2][1] == 4 and isinstance(wheres[where_idx_1][3],list) and isinstance(wheres[where_idx_2][3],list) and wheres[where_idx_1][3][1].split(".")[1] == wheres[where_idx_2][3][1].split(".")[1]:
            wheres[middle] = "intersect_"
        elif wheres[where_idx_1][1] == 4 and wheres[where_idx_2][1] == 3 and isinstance(wheres[where_idx_1][3],list) and isinstance(wheres[where_idx_2][3],list) and wheres[where_idx_1][3][1].split(".")[1] == wheres[where_idx_2][3][1].split(".")[1]:
            wheres[middle] = "intersect_"
        #################################
        elif wheres[where_idx_1][1] not in [8,12,15] and wheres[where_idx_2][1] == 15 and wheres[0][1] not in [8,12,15]:
            wheres[middle] = "intersect_"
        #################################
        elif wheres[where_idx_1][1] == 3 and wheres[where_idx_2][1] == 4 and (wheres[where_idx_1][3]) >= (wheres[where_idx_2][3]) and wheres[where_idx_1][2][1][1] == wheres[where_idx_2][2][1][1]:
            wheres[middle] = "intersect_"
        elif wheres[where_idx_1][1] == 4 and wheres[where_idx_2][1] == 3 and (wheres[where_idx_1][3]) <= (wheres[where_idx_2][3]) and wheres[where_idx_1][2][1][1] == wheres[where_idx_2][2][1][1]:
            wheres[middle] = "intersect_"
        elif wheres[where_idx_1][1] == 5 and wheres[where_idx_2][1] == 6 and (wheres[where_idx_1][3]) > (wheres[where_idx_2][3]) and wheres[where_idx_1][2][1][1] == wheres[where_idx_2][2][1][1]:
            wheres[middle] = "intersect_"
        elif wheres[where_idx_1][1] == 6 and wheres[where_idx_2][1] == 5 and (wheres[where_idx_1][3]) < (wheres[where_idx_2][3]) and wheres[where_idx_1][2][1][1] == wheres[where_idx_2][2][1][1]:
            wheres[middle] = "intersect_"
        elif wheres[where_idx_1][1] == 4 and wheres[where_idx_2][1] == 5 and (wheres[where_idx_1][3]) < (wheres[where_idx_2][3]) and wheres[where_idx_1][2][1][1] == wheres[where_idx_2][2][1][1]:
            wheres[middle] = "intersect_"
        elif wheres[where_idx_1][1] == 5 and wheres[where_idx_2][1] == 4 and (wheres[where_idx_1][3]) > (wheres[where_idx_2][3]) and wheres[where_idx_1][2][1][1] == wheres[where_idx_2][2][1][1]:
            wheres[middle] = "intersect_"
        elif wheres[where_idx_1][1] == 3 and wheres[where_idx_2][1] == 6 and (wheres[where_idx_1][3]) > (wheres[where_idx_2][3]) and wheres[where_idx_1][2][1][1] == wheres[where_idx_2][2][1][1]:
            wheres[middle] = "intersect_"
        elif wheres[where_idx_1][1] == 6 and wheres[where_idx_2][1] == 3 and (wheres[where_idx_1][3]) < (wheres[where_idx_2][3]) and wheres[where_idx_1][2][1][1] == wheres[where_idx_2][2][1][1]:
            wheres[middle] = "intersect_"
        return wheres
    try:
        if ((len(wheres) == 3 and wheres[1] == "and") or (len(wheres) == 5 and wheres[1] == "and" and wheres[3] == "and")) and (wheres[0][2][1][1] == wheres[2][2][1][1] or (".*" in wheres[0][2][1][1] and ".*" in wheres[2][2][1][1]) or wheres[2][2][1][0] != 0 ):
            wheres = replace_str_for_inference(wheres,0,2,sq)   
        if (len(wheres) == 5 and wheres[1] == "and" and wheres[3] == "and") and (wheres[4][2][1][1] == wheres[2][2][1][1] or (".*" in wheres[4][2][1][1] and ".*" in wheres[2][2][1][1]) or wheres[4][2][1][0] != 0):
            if wheres[4][2] == wheres[2][2] and wheres[4][1] == wheres[2][1] and wheres[0][1] == 2:
                wheres.insert(4,wheres[0])
                wheres.insert(5,"and")
                wheres = replace_str_for_inference(wheres,2,6,sq,middle=3)
            else:
                wheres = replace_str_for_inference(wheres,2,4,sq)
        elif (len(wheres) == 7 and wheres[1] == "and" and wheres[3] == "and" and wheres[5] == "and") and not(wheres[0][1] == wheres[4][1] and wheres[0][1] in [8,12] and wheres[0][3][1] != wheres[4][3][1]) and ((wheres[0][2][1][1] == wheres[4][2][1][1] and wheres[2][2][1][1] == wheres[6][2][1][1]) or (wheres[2][2][1][1] == wheres[4][2][1][1] and wheres[0][2][1][1] == wheres[6][2][1][1])):
            if wheres[2][2][1][1] == wheres[6][2][1][1]:
                wheres = replace_str_for_inference(wheres,2,6,sq,middle=3)
            else:
                wheres = replace_str_for_inference(wheres,0,6,sq)
    except:
        pass
    return wheres



def union_inference(wheres,sq,table_json,top_select_table_list):
    try:
        if len(wheres) == 5 and wheres[1] == "or" and wheres[3] == "and":
            wheres[1] = "union_"
        elif len(wheres) == 7 and wheres[1] == "and" and wheres[3] == "or" and wheres[5] == "and" and ((wheres[0][2][1] ==  wheres[4][2][1] and wheres[2][2][1] ==  wheres[6][2][1]) or (wheres[0][2][1] ==  wheres[6][2][1] and wheres[2][2][1] ==  wheres[4][2][1])):
            wheres[3] = "union_"
        elif len(wheres) == 5 and wheres[1] == "and" and wheres[3] == "or" and not (wheres[0][1] == 15 and wheres[2][2][1][1].split(".")[0] == wheres[4][2][1][1].split(".")[0]): #and (((wheres[4][2][1][0] != 0 or wheres[2][2][1][0] != 0) and (wheres[4][2][1][1] != wheres[2][2][1][1] or ".*" in wheres[4][2][1][1] or ".*" in wheres[2][2][1][1])) or wheres[4][1] == 15):
            wheres[3] = "union_"
        elif len(wheres) == 5 and wheres[1] == "or" and wheres[0][2][1][1] == "@.@":
            wheres[1] = "union_"
        elif len(wheres) == 5 and wheres[3] == "or" and wheres[2][2][1][1] == "@.@":
            wheres[3] = "union_"
        elif len(wheres) == 3 and wheres[1] == "or" and (((wheres[0][2][1][0] != 0 or wheres[2][2][1][0] != 0) and (wheres[0][2][1][1] != wheres[2][2][1][1] or ".*" in wheres[0][2][1][1] or ".*" in wheres[2][2][1][1])) or wheres[2][1] == 15):
            wheres[1] = "union_"
        elif len(wheres) == 3 and wheres[1] == "or" and wheres[0][2][1][1] == "@.@" and not (wheres[2][2][1][1] == "@.@" and wheres[0][1] == wheres[2][1] and wheres[0][3][1].split(".")[0] == wheres[2][3][1].split(".")[0]):
            wheres[1] = "union_"
        elif len(wheres) == 3 and wheres[1] == "or" and wheres[2][1] == 2 and wheres[0][1] not in [8,12,15]:
            left = wheres[0][2][1][1].split(".")
            right = wheres[2][2][1][1].split(".")
            if right[0] not in top_select_table_list and left[0] != right[0]:
                top_select_table_list = copy.deepcopy(top_select_table_list)
                if left[0] not in top_select_table_list:
                    top_select_table_list.append(left[0])
                from_table_net,table_fk_list = get_table_network(table_json, top_select_table_list, None)
                top_select_table_list = []
                for tb in from_table_net:
                    if tb:
                        top_select_table_list.append(table_json['table_names_original'][tb[0]])
                if right[0] not in top_select_table_list:
                    wheres[1] = "union_"
        elif len(wheres) == 7 and wheres[3] == "or" and wheres[5] == "and" and not(wheres[0][1] == wheres[4][1] and wheres[0][1] in [8,12] and wheres[0][3][1] != wheres[4][3][1]):
            wheres[3] = "union_"
    except:
        pass
    return wheres



def except_inference(wheres,sq):
    try:
        if ((len(wheres) == 3 and wheres[1] == "and") or (len(wheres) == 5 and wheres[1] == "and" and wheres[3] == "and")) and wheres[2][1] == 7 and not (wheres[0][2][1][1] == wheres[2][2][1][1] and wheres[0][1] == 7):
            if (wheres[0][2][1][1] == wheres[2][2][1][1] and wheres[0][1] == 2) or (wheres[0][1] == 7) or wheres[0][2][1][0]:
                wheres[1] = "except_"
                wheres[2][1] = 2
        if (len(wheres) == 5 and wheres[1] == "and" and wheres[3] == "and") and wheres[4][1] == 7:
            if (wheres[0][1] in [8,12] and wheres[2][1] == 2) or (wheres[0][1] == 2 and wheres[2][1] == 2) or  wheres[2][2][1][0]:
                wheres[3] = "except_"
                wheres[4][1] = 2
        elif (len(wheres) == 5 and wheres[1] == "and" and wheres[3] == "and") and wheres[0][1] == 2 and wheres[2][1] == 7:
            wheres[1] = "except_"
            wheres[2][1] = 2
            if wheres[4][1] == 7:
                wheres[4][1] = 2
        elif (len(wheres) == 7 and wheres[1] == "and" and wheres[3] == "and" and wheres[5] == "and") and wheres[0][1] in [15] and wheres[2][1] == 2 and wheres[4][1] == 7:
            wheres[3] = "except_"
            wheres[4][1] = 2
            if wheres[6][1] == 7:
                wheres[6][1] = 2
        elif (len(wheres) == 5 and wheres[1] == "and" and wheres[3] == "or") and wheres[0][1] not in [8,12,15] and (wheres[2][1] == 7 or wheres[4][1] == 7) and (wheres[0][2][1][1].split(".")[0] != wheres[2][2][1][1].split(".")[0] or wheres[0][2][1][1].split(".")[0] != wheres[4][2][1][1].split(".")[0]):
            wheres[1] = "except_"
            wheres[2][1] = 2
            if wheres[4][1] == 7:
                wheres[4][1] = 2
        elif (len(wheres) == 7 and wheres[1] == "and" and wheres[3] == "and" and wheres[5] == "and") and wheres[0][1] not in [8,12,15] and wheres[4][1] == 7:
                wheres[3] = "except_"
                wheres[4][1] = 2
                if wheres[6][1] == 7:
                    wheres[6][1] = 2
    except:
        pass
    return wheres



def infer_group_for_exact_match(group_list,table_json,table_list,num_select,sq=None,agg_tables=[],from_table_net=[]):
    """V1.2
    infer the intersect
    """
    select_no_agg_tables = set()
    for g in group_list:
        select_no_agg_tables.add([n.lower() for n in table_json['table_names_original']].index(g.split(".")[0].lower()))

    if len(group_list) > 1:
        if not from_table_net:
            from_table_net = [[],[]]
        net_fks = [fk for fks in from_table_net[0] for fk in fks]

        if len(select_no_agg_tables) == 1:
            t_idx = list(select_no_agg_tables)[0]
            pk = primary_keys(table_json,t_idx,pk_only=True)
            if pk >= 0:
                return [table_json['table_column_names_original'][pk][1]]
            for col in group_list:
                col_idx = table_json['tc_fast'].index(col.lower())
                if col_idx in net_fks:
                    return [col]
            for col in group_list:
                if "name" in col and "firstname" not in col and "first_name" not in col and "lastname" not in col and "last_name" not in col and "forename" not in col and "surname" not in col:
                    return [col]
        elif len(select_no_agg_tables) >= 2 and len(table_list) >= 2:
            from_table_net,table_fk_list = get_table_network(table_json, table_list, None, group_list=group_list)
            if from_table_net:
                return filter_PK(from_table_net,agg_tables,table_json,select_no_agg_tables)
    elif sq and num_select == 1 and len(table_list) >= 2 and group_list and "lname" not in group_list[0] and "fname" not in group_list[0] and "lastname" not in group_list[0] and "firstname" not in group_list[0] and "l_name" not in group_list[0] and "f_name" not in group_list[0] and "last_name" not in group_list[0] and "first_name" not in group_list[0]:
        if sq.sub_sequence_type.count(1) == 1 and 0 not in sq.sub_sequence_type and sq.sub_sequence_type[0] == 1 and sq.pattern_tok[0][-1] in ["SC","STC","COL","BCOL","TABLE-COL"]:
            pass
        elif len(sq.sub_sequence_type) == 1 and "TABLE" not in sq.pattern_tok[0] and "ST" not in sq.pattern_tok[0]:
            pass
        else:
            from_table_net,table_fk_list = get_table_network(table_json, table_list, None, group_list=group_list)
            if from_table_net:
                return filter_PK(from_table_net,agg_tables,table_json,select_no_agg_tables)
    elif num_select == 1 and len(table_list) >= 2: 
        if group_list and "name" not in group_list[0] and "title" not in group_list[0]:
            return group_list
        from_table_net,table_fk_list = get_table_network(table_json, table_list, None, group_list=group_list)
        if from_table_net:
            return filter_PK(from_table_net,agg_tables,table_json,select_no_agg_tables)
    elif not group_list and table_list:
        for ta in table_list:
            ta = ta.lower()
            for i,t2 in enumerate(table_json['table_names_original']):
                if t2.lower() == ta:
                    for j, tc in enumerate(table_json['table_column_names_original']):
                        if tc[0] == i:
                            for fk in table_json['foreign_keys']:
                                if j in fk or j in table_json['primary_keys']:
                                    return [tc[1]]
                    break
    return group_list


def filter_PK(from_table_net,agg_tables,table_json,select_no_agg_tables):
    re_col = 0
    if agg_tables:

        for fk in from_table_net[0]:
            if (fk[0] in table_json['primary_keys'] and table_json['column_names'][fk[0]][0] in agg_tables[1] ) or ( fk[1] in table_json['primary_keys'] and table_json['column_names'][fk[1]][0] in agg_tables[1] ):
                continue
            if not re_col and table_json['column_names'][fk[0]][0] in select_no_agg_tables:
                re_col = fk[0]
            elif not re_col and table_json['column_names'][fk[1]][0] in select_no_agg_tables:
                re_col = fk[1]
            if table_json['column_names'][fk[0]][0] in select_no_agg_tables or table_json['column_names'][fk[1]][0] in select_no_agg_tables:
                if table_json['column_names'][fk[0]][0] in agg_tables[1]:
                    return [table_json['table_column_names_original'][fk[0]][1]]
                if table_json['column_names'][fk[1]][0] in agg_tables[1]:
                    return [table_json['table_column_names_original'][fk[1]][1]]
                if not re_col and table_json['column_names'][fk[0]][0] in select_no_agg_tables:
                    re_col = fk[0]
                elif not re_col and table_json['column_names'][fk[1]][0] in select_no_agg_tables:
                    re_col = fk[1]
    return [table_json['tc_fast'][re_col]] if re_col else [table_json['tc_fast'][from_table_net[0][0][0]]]

def get_agg_tables(sql_dict,table_json):
    agg_tables = [[],[]]
    for sel in sql_dict['select'][1]:
        if (sel[1][1][0] or sel[0]):
            a_table,a_col = sel[1][1][1].lower().split(".")
            agg_tables[0].append(a_table)
            agg_tables[1].append(table_json['column_names'][table_json["tc_fast"].index(sel[1][1][1].lower())][0])
    sub_query = False
    for w in sql_dict['where']:
        if type(w) == list and w[2][1][0]:
            a_table,a_col = w[2][1][1].lower().split(".")
            agg_tables[0].append(a_table)
            agg_tables[1].append(table_json['column_names'][table_json["tc_fast"].index(w[2][1][1].lower())][0])
            if type(w) == list and type(w[3]) == list:
                sub_query = True
                break
    if not sub_query and sql_dict["orderBy"] and sql_dict["orderBy"][1][0][1][0]:
        a_table,a_col = sql_dict["orderBy"][1][0][1][1].lower().split(".")
        agg_tables[0].append(a_table)
        agg_tables[1].append(table_json['column_names'][table_json["tc_fast"].index(sql_dict["orderBy"][1][0][1][1].lower())][0])
    return agg_tables


def group_back_because_eva_bug_in_spider(groupby_list,table_json,agg_tables,or_groupby_list,from_table_net):
    if len(groupby_list) == 1:
        if not from_table_net:
            from_table_net = [[],[]]
        net_fks = [fk for fks in from_table_net[0] for fk in fks]
        g_table,g_col = groupby_list[0].lower().split('.')
        g_col_idx = table_json["tc_fast"].index(groupby_list[0].lower())
        for t_name,t_idx in zip(agg_tables[0],agg_tables[1]):
            if g_table == t_name:
                return groupby_list
            for fk in table_json['foreign_keys']:
                if g_col_idx in fk and ((fk[0] == g_col_idx and table_json['column_names'][fk[1]][0] == t_idx) or (fk[1] == g_col_idx and table_json['column_names'][fk[0]][0] == t_idx)):
                    if or_groupby_list and len(or_groupby_list) == 1 and ((fk[0] == g_col_idx and fk[1] in table_json["primary_keys"]) or (fk[0] != g_col_idx and fk[0] in table_json["primary_keys"])):
                        return or_groupby_list
                    return [table_json["tc_fast"][fk[1]]] if fk[0] == g_col_idx else [table_json["tc_fast"][fk[0]]]
            if g_col not in ['id','ids','name','names'] and g_table != t_name and t_name + "." + g_col in table_json["tc_fast"] and table_json["tc_fast"].index(t_name + "." + g_col) in net_fks:
                return [t_name + "." + g_col]
    return groupby_list

def infer_IUE_select_col(sql_dict,table_json,break_idx,top_sql_list,top_select_table_list):
    if "@" in sql_dict['where'][break_idx+1][2][1][1]:
        start_table = sql_dict['where'][break_idx+1][3][1][:-2].lower()
    else:
        start_table = sql_dict['where'][break_idx+1][2][1][1].split(".")[0].lower()

    table_list = [start_table]
    old_select = sql_dict['select'][1][0][1][1][1].lower()
    [top_select_t,top_select_c] = old_select.split(".")
    if top_select_t == start_table:
        return " "  + sql_dict['where'][break_idx][:-1] +  " " + top_sql_list[0], top_select_table_list
    else:
        sub_sql = " "  + sql_dict['where'][break_idx][:-1] +  " select "                        
        old_t_idx = -1
        new_t_idx = -1
        old_c_idx = -1
        for i, tbl in enumerate(table_json['table_names_original']):
            if tbl.lower() == table_list[0]:
                new_t_idx = i
            if tbl.lower() == top_select_t:
                old_t_idx = i
        if old_t_idx == -1 or new_t_idx == -1 or old_t_idx == new_t_idx:
            return " "  + sql_dict['where'][break_idx][:-1] +  " " + top_sql_list[0], top_select_table_list
        
        if len(sql_dict['select'][1]) > 1:
            if table_json['table_names_original'][new_t_idx].lower() not in top_select_table_list:
                top_select_table_list += [table_json['table_names_original'][new_t_idx].lower()]
            return " "  + sql_dict['where'][break_idx][:-1] +  " " + top_sql_list[0], top_select_table_list
        
        for i,col in enumerate(table_json['table_column_names_original']):
            if old_select == col[1].lower() and old_t_idx == col[0]:
                old_c_idx = i
                break
        if old_c_idx == -1:
            return " "  + sql_dict['where'][break_idx][:-1] +  " " + top_sql_list[0], top_select_table_list

        # find foreign key
        other_fk = -1
        for fk in table_json['foreign_keys']:
            if old_c_idx in fk:
                other_fk_tmp = fk[0] if old_c_idx == fk[1] else fk[1]
                if table_json['column_names_original'][other_fk_tmp][0] == new_t_idx:
                    if other_fk == -1:
                        other_fk = other_fk_tmp
                    elif table_json['column_names_original'][old_c_idx][1] == table_json['column_names_original'][other_fk_tmp][1]:
                        other_fk = other_fk_tmp
                    elif other_fk_tmp in table_json['original_primary_keys']:
                        other_fk = other_fk_tmp
        if other_fk > 0:
            return (sub_sql + table_json['table_column_names_original'][other_fk][1]).lower(), [table_json['table_names_original'][new_t_idx].lower()]
        
        # find same name column
        final_choice = None
        for o_col,col in zip(table_json['column_names_original'],table_json['column_names']):
            if o_col[0] == new_t_idx and (o_col[1] == table_json['column_names_original'][old_c_idx][1] or col[1] == table_json['column_names'][old_c_idx][1] or col[1] == table_json['table_names'][table_json['column_names'][old_c_idx][0]]+" " + table_json['column_names'][old_c_idx][1]):
                if o_col[1] == '*':
                    return " "  + sql_dict['where'][break_idx][:-1] +  " " + top_sql_list[0], [table_json['table_names_original'][new_t_idx].lower()]
                else:
                    final_choice = (sub_sql + table_json['table_names_original'][new_t_idx] + "." + o_col[1]).lower(), [table_json['table_names_original'][new_t_idx].lower()]
                    if col[1] not in ["name","names","id","ids"]:
                        return final_choice
        if final_choice:
            return final_choice
        return " "  + sql_dict['where'][break_idx][:-1] +  " " + top_sql_list[0], top_select_table_list + [table_json['table_names_original'][new_t_idx].lower()]



def inference_sql(sql_dict, table_json, args, join_on_label=None, sq=None, return_with_join_on = False):
    re_generate,re_sql,_,__ = search_all_join_on(sql_dict, table_json, args, join_on_label, sq, True)
    if re_generate:
        if re_generate == 1:
            args_ = copy.deepcopy(args)
            args_.iue2subquery = False
        elif re_generate == 2:
            args_ = copy.deepcopy(args)
            args_.join2subquery = False
        elif re_generate == 3:
            args_ = copy.deepcopy(args)
            args_.groupby2subquery = False
        else:
            args_ = args
        re_generate,re_sql,_,__ = search_all_join_on(re_sql, table_json, args_, join_on_label, sq, False)
    if return_with_join_on:
        return re_sql,_,__
    return re_sql



def from_net_to_str(table_json, from_table_netss):
    str_all = []
    for fts in from_table_netss:
        str_list = []
        for ft in fts:
            str_list.append(create_from_table(ft,table_json['table_names_original'], table_json['table_column_names_original'], []))
        str_all.append(str_list)
    return str_all

def extract_select_columns(sub_sql):
    sub_sql_tokens = sub_sql.split(" ")
    see_select = False
    re_columns = ""
    for i,tok in enumerate(sub_sql_tokens):
        if see_select:
            if "." not in tok or "(" in tok:
                break
            re_columns += tok + " "
        if tok.lower() == "select":
            see_select = True
    return re_columns

def fk_replace_IUE(sql_dict,table_json,break_idx, top_sql_list,top_select_table_list):
    if len(sql_dict['select'][1]) == 1 and ((break_idx == 0 and len(sql_dict['where']) > 3) or (break_idx > 0 and len(sql_dict['where']) > break_idx+1)):
        if break_idx == 0:
            next_con = 3
        else:
            next_con = break_idx + 1
        col_name = sql_dict['select'][1][0][1][1][1].lower()
        col_idx = 0
        col_name2 = sql_dict['where'][next_con][2][1][1].lower()
        col_idx2 = 0

        for i,col in enumerate(table_json['table_column_names_original']):
            if col[1].lower() == col_name2:
                col_idx2 = i
                break

        for i,col in enumerate(table_json['table_column_names_original']):
            if col[1].lower() == col_name:
                for fk in table_json['foreign_keys']:
                    if (i == fk[0] and table_json['column_names_original'][fk[1]][0] == table_json['column_names_original'][col_idx2][0]) or (i == fk[1] and table_json['column_names_original'][fk[0]][0] == table_json['column_names_original'][col_idx2][0]):
                        col_idx = i
                        break
                if col_idx2 and not col_idx and table_json['column_names'][i][1].lower() not in ["name","names","id","ids"] and break_idx > 0 and break_idx + 1 < len(sql_dict['where']) and isinstance(sql_dict['where'][break_idx-1],list) and isinstance(sql_dict['where'][break_idx+1],list) and sql_dict['where'][break_idx-1][2][1][1].split(".")[0] != sql_dict['where'][break_idx+1][2][1][1].split(".")[0]:
                    fk_or_pk = False
                    if i in table_json['primary_keys']:
                        fk_or_pk = True
                    else:
                        for fk in table_json['foreign_keys']:
                            if i in fk:
                                fk_or_pk = True
                    if not fk_or_pk:
                        pure_col_name = col_name.split(".")[1]
                        for j,col in enumerate(table_json['column_names_original']):
                            if col[0] == table_json['column_names_original'][col_idx2][0] and col[1].lower() == pure_col_name and table_json['column_names'][i][1].lower() == table_json['column_names'][j][1].lower():
                                col_idx = i
                                break
                elif table_json['column_names'][i][1].lower() in ["name","names","id","ids"] :
                    fk_or_pk = False
                    for fk in table_json['foreign_keys']:
                        if i in fk:
                            fk_or_pk = True
                    if not fk_or_pk:
                        pure_col_name = (table_json['table_names'][table_json['column_names'][i][0]] + " " +  table_json['column_names'][i][1]).lower()
                        for j,col in enumerate(table_json['column_names']):
                            if col[0] == table_json['column_names'][col_idx2][0] and ( col[1].lower() == pure_col_name or table_json['column_names'][i][1].lower() == (table_json['table_names'][col[0]] + " " + col[1]).lower() ):
                                col_idx = i
                                break
                break
        if col_idx:
            return infer_IUE_select_col(sql_dict,table_json,break_idx,top_sql_list,top_select_table_list)

    
    return " "  + sql_dict['where'][break_idx][:-1] +  " " + top_sql_list[0], top_select_table_list


def search_bcol(sq,table,where,sql_dict,where_i):
    """
    There is no boolean value in the question
    """
    where_col_idx = where[2][1][1] if where[2][1][1] not in table['tc_fast']  else table['tc_fast'].index(where[2][1][1])
    for (i_t,type_), pts in zip(enumerate(sq.sub_sequence_type),sq.pattern_tok):
        if "BCOL" in pts:
            for j,pt in enumerate(pts):
                if pt == "BCOL":
                    for col in sq.col_match[i_t][j]:
                        col_id = reversed_link_back_col(col[0], table)
                        if col_id == where_col_idx:
                            if "NOT" not in pts or (where_i > 1 and type(sql_dict["where"][where_i-2]) == list and sql_dict["where"][where_i-2][1] == 12) or (where_i > 3 and type(sql_dict["where"][where_i-4]) == list and sql_dict["where"][where_i-4][1] == 12):
                                for v in table['data_samples'][where_col_idx]:
                                    if (type(v) == str and v.lower() in ["y","yes","true","t","1","good","g","success","s"]) or (type(v) == int and v == 1):
                                        where[3] = "'"+v+"'" if type(v) == str else v
                                        break
                            else:
                                for v in table['data_samples'][where_col_idx]:
                                    if (type(v) == str and v.lower() in ["n","no","false","f","0","bad","b","fail","f"]) or (type(v) == int and v == 0):
                                        where[3] = "'"+v+"'" if type(v) == str else v
                                        break
                        if where[3] != '"terminal"':
                            break
                if where[3] != '"terminal"':
                    break
        if where[3] != '"terminal"':
            break
    if where[3] == '"terminal"':
        for v in table['data_samples'][where_col_idx]:
            if " " + str(v).lower() + " " in " " + sq.question_or.lower() + " ":
                where[3] = "'"+v+"'" if type(v) == str else v
                break
    if where[3] == '"terminal"':
        if len(table['data_samples'][where_col_idx]) > 0:
            for v in table['data_samples'][where_col_idx]:
                if (type(v) == str and v.lower() in ["y","yes","true","t","1","good","g","success","s"]) or (type(v) == int and v == 1):
                    where[3] = "'"+v+"'" if type(v) == str else v
                    break
    if where[3] == '"terminal"' and table['data_samples'][where_col_idx]:
        where[3] = "'"+table['data_samples'][where_col_idx][0]+"'" if type(table['data_samples'][where_col_idx][0]) == str else table['data_samples'][where_col_idx][0]


def search_db(sq,table,where,used_value,sq_sub_idx=-1):
    where_col_idx = where[2][1][1] if where[2][1][1] not in table['tc_fast']  else table['tc_fast'].index(where[2][1][1])

    for v in table['data_samples'][where_col_idx]:
        if where[3] == '"terminal"' and " " + str(v).lower() + " " in " " + sq.question_or.lower() + " ":
            vl = str(v).lower()
            for i, sts in enumerate(sq.sub_sequence_toks):
                if sq_sub_idx >= 0 and sq_sub_idx != i:
                    continue
                if where[3] == '"terminal"':
                    for j, st in enumerate(sts):
                        if [i,j] not in used_value and st.lower() == vl:
                            where[3] = "'"+v+"'" if type(v) == str else v
                            used_value.append([i,j])
                            return True
    
    if where[3] == '"terminal"':
        for v in table['data_samples'][where_col_idx]:
            if where[3] == '"terminal"' and type(v) == str :
                vl = v.lower()
                for i, sts in enumerate(sq.sub_sequence_toks):
                    if sq_sub_idx >= 0 and sq_sub_idx != i:
                        continue
                    if where[3] == '"terminal"':
                        for j, st in enumerate(sts):
                            if len(st) >= 3 and (([i,j] not in used_value and " " not in v and (st.lower().startswith(vl) or vl.startswith(st.lower()))) or (" " in v and [i,j] not in used_value and vl.startswith(st.lower()+" ") and "DB" in sq.pattern_tok[i][j])):
                                where[3] = "'"+v+"'"
                                used_value.append([i,j])
                                return True
                            elif " " in v and j + 1 < len(sts) and [i,j] not in used_value and [i,j+1] not in used_value and len(sts[j+1]) > 2 and (vl.startswith(st.lower()+" "+sts[j+1]+" ") or vl.startswith(st.lower()+" "+sts[j+1][:-1]+" ") or vl == st.lower()+" "+sts[j+1] or vl == st.lower()+" "+sts[j+1][:-1]):
                                where[3] = "'"+v+"'"
                                used_value.append([i,j])
                                used_value.append([i,j+1])
                                return True
                           
    return False

def no_more_num_cond(wheres,i,table_json):
    for w_i,where in enumerate(wheres):
        if w_i > i and type(where) == list and ((where[2][1][1] in table_json['tc_fast'] and table_json['column_types'][table_json['tc_fast'].index(where[2][1][1])] in ["number","year","time"]) or ( type(where[2][1][1]) == int and table_json['column_types'][where[2][1][1]] in ["number","year","time"])):
            return False
    return True


def get_num_for_limit(pts,sq,sql_dict,i,table_json):
    num_idx = pts.index("NUM")
    num = str2num(sq.sub_sequence_toks[i][num_idx])
    if num == 1 and pts.count("NUM") > 1:
        num_idx = pts.index("NUM",1+num_idx)
        num = str2num(sq.sub_sequence_toks[i][num_idx])
    if num > 10 and sql_dict['where']:
        for where in sql_dict['where']:
            if type(where) == list and ((where[2][1][1] in table_json['tc_fast'] and table_json['column_types'][table_json['tc_fast'].index(where[2][1][1])] in ["number","year","time"]) or ( type(where[2][1][1]) == int and table_json['column_types'][where[2][1][1]] in ["number","year","time"])):
                num = 0
    return num_idx,num


def generate_where_values(i,pts,sq,used_value,patterns,where,table_json,sql_dict,where_idx):
    def add_one_value(pt,pt_target,pts,sq,used_value,i,j):
        re_num = sq.sub_sequence_toks[i][j]
        used_value.append([i,j])
        if pt == pt_target and j + 1 < len(pts) and pts[j+1] == pt_target and (pt_target != "DB" or sq.db_match[i][j] == sq.db_match[i][j+1]):
            re_num += (" " + sq.sub_sequence_toks[i][j+1])
            used_value.append([i,j+1])
            if  j + 2 < len(pts) and pts[j+2] == pt_target and (pt_target != "DB" or sq.db_match[i][j+2] == sq.db_match[i][j+1]):
                re_num += (" " + sq.sub_sequence_toks[i][j+2])
                used_value.append([i,j+2])
                if  j + 3 < len(pts) and pts[j+3] == pt_target  and (pt_target != "DB" or sq.db_match[i][j+3] == sq.db_match[i][j+2]):
                    re_num += (" " + sq.sub_sequence_toks[i][j+3])
                    used_value.append([i,j+3])
                    if  j + 4 < len(pts) and pts[j+4] == pt_target and (pt_target != "DB" or sq.db_match[i][j+4] == sq.db_match[i][j+3]):
                        re_num += (" " + sq.sub_sequence_toks[i][j+4])
                        used_value.append([i,j+4])
                        if  j + 5 < len(pts) and pts[j+5] == pt_target and (pt_target != "DB" or sq.db_match[i][j+5] == sq.db_match[i][j+4]):
                            re_num += (" " + sq.sub_sequence_toks[i][j+5])
                            used_value.append([i,j+5])
            re_num = "'" + re_num + "'"
        else:
            re_num = "'" + re_num + "'"
        return re_num,used_value
    def search_like(sq,used_value,i,j,sql_dict,where_idx,recurrent=False):
        re_w = None
        for (w_j,w),pt in zip(enumerate(sq.sub_sequence_toks[i]), sq.pattern_tok[i]):
            if re_w:
                if [i,w_j] not in used_value and (pt in ["PDB","UDB","SDB","DB"] or (w.isalpha() and not w.islower())):
                    re_w += (" " + w)
                    used_value.append([i,w_j])
                else:
                    break
            elif [i,w_j] not in used_value and (pt in ["PDB","UDB","SDB","DB"] or (w.isalpha() and not w.islower() and w_j != 0)):
                re_w = w
                used_value.append([i,w_j])
        if recurrent:
            return re_w
        if not re_w and len(sql_dict["where"]) == 3 and  where_idx == 2 and i + 1 < len(sq.sub_sequence_toks):
            re_w = search_like(sq,used_value,i+1,0,sql_dict,where_idx,recurrent=True)

        if not re_w and not recurrent:
            for (w_j,w),pt in zip(enumerate(sq.sub_sequence_toks[i]), sq.pattern_tok[i]):
                if w_j <= j:
                    continue
                if [i,w_j] in used_value or pt in ["ST","STC","SC","COL","TABLE","TABLE-COL"]:
                    continue
                if w not in ["word","substring","tilt","tilted","tilting","letter","character"] and (w.isupper() or pt not in ["IN","#"]):
                    re_w = w
                    used_value.append([i,w_j])
                    break
                elif w_j > 0 and sq.sub_sequence_toks[i][w_j-1] in ["word","substring","letter","character"] and w.isalpha():
                    if w == "the" and w_j + 1 < len(sq.sub_sequence_toks[i]) and sq.sub_sequence_toks[i][w_j+1].isalpha():
                        w = sq.sub_sequence_toks[i][w_j+1]
                        w_j += 1
                    re_w = w
                    used_value.append([i,w_j])
                    break
        if re_w:
            if " start with " in sq.question_or or " starting with " in sq.question_or or " started with " in sq.question_or or " starts with " in sq.question_or or " begin with " in sq.question_or or " beginning with " in sq.question_or or " began with " in sq.question_or or " begins with " in sq.question_or or " start from " in sq.question_or or " starts from " in sq.question_or or " starting from " in sq.question_or or " started from " in sq.question_or or " begin from " in sq.question_or or " begins from " in sq.question_or or " began from " in sq.question_or or " beginning from " in sq.question_or:
                re_w = re_w + "%"
            elif " end with " in sq.question_or or " ending with " in sq.question_or or " ended with " in sq.question_or or " ends with " in sq.question_or or " end by " in sq.question_or or " ending by " in sq.question_or or " ended by " in sq.question_or or " ends by " in sq.question_or:
                re_w = "%" + re_w 
            re_w = "'"+re_w+"'"
            return True,re_w,used_value
        return False,re_w,used_value
    for j,pt in enumerate(pts):
        if patterns == ["NUM","DATE","YEAR"] and pt in ["NUM","DATE","YEAR"] and [i,j] not in used_value:
            if pt == "DATE":
                re_num,used_value = add_one_value(pt,"DATE",pts,sq,used_value,i,j)
            else:
                re_num = sq.sub_sequence_toks[i][j]
                re_num = str2num(re_num)
                used_value.append([i,j])
                if sql_dict["where"][where_idx][1] != 1 and re_num == 1 and "NUM" in pts[j+1:] and [i,pts.index("NUM",j+1)] not in used_value and no_more_num_cond(sql_dict["where"],where_idx,table_json):
                    continue
                elif sql_dict["where"][where_idx][1] != 1 and re_num == 1 and "YEAR" in pts[j+1:] and [i,pts.index("YEAR",j+1)] not in used_value and no_more_num_cond(sql_dict["where"],where_idx,table_json):
                    continue
                elif sql_dict["where"][where_idx][1] != 1 and re_num == 1 and "DB" in pts and [i,pts.index("DB")] not in used_value and where_idx + 1 >= len(sql_dict["where"]) and sq.sub_sequence_toks[i][pts.index("DB")].isdigit():
                    return False,re_num,used_value
                elif sql_dict["where"][where_idx][1] != 1 and re_num == 1 and i + 1 < len(sq.pattern_tok) and "NUM" in sq.pattern_tok[i+1] and [i+1,sq.pattern_tok[i+1].index("NUM")] not in used_value and where_idx + 1 >= len(sql_dict["where"]):
                    return False,re_num,used_value
                elif sql_dict["where"][where_idx][1] != 1 and re_num == 1 and i + 1 < len(sq.pattern_tok) and "YEAR" in sq.pattern_tok[i+1] and [i+1,sq.pattern_tok[i+1].index("YEAR")] not in used_value and where_idx + 1 >= len(sql_dict["where"]):
                    return False,re_num,used_value
                elif sql_dict["where"][where_idx][1] != 1 and re_num == 1 and i + 1 < len(sq.pattern_tok) and "DB" in sq.pattern_tok[i+1] and [i+1,sq.pattern_tok[i+1].index("DB")] not in used_value and where_idx + 1 >= len(sql_dict["where"]) and sq.sub_sequence_toks[i+1][sq.pattern_tok[i+1].index("DB")].isdigit():
                    return False,re_num,used_value
            return True,re_num,used_value
        elif patterns == ["DB","SDB"] and pt == "DB" and [i,j] not in used_value and str_is_num(sq.sub_sequence_toks[i][j]):
            re_num = str2num(sq.sub_sequence_toks[i][j]) 
            used_value.append([i,j])
            return True,re_num,used_value
        elif patterns == ["DB","SDB"] and pt == "DB" and [i,j] not in used_value and str_is_date(sq.sub_sequence_toks[i][j]):
            re_num,used_value = add_one_value(pt,pt,pts,sq,used_value,i,j)
            return True,re_num,used_value
        elif patterns == ["DB","SDB"] and pt == "SDB" and [i,j] not in used_value and sq.sub_sequence_toks[i][j].lower() in  ['none', 'nil','null','inf','-inf','+inf']:
            re_num = "'" + sq.sub_sequence_toks[i][j] + "'"
            used_value.append([i,j])
            return True,re_num,used_value
        elif patterns == ["NOT"] and pt == "NOT" and [i,j] not in used_value:
            re_num = "'null'"
            used_value.append([i,j])
            return True,re_num,used_value
        elif patterns == ["TEXT_DB","DB"] and pt == "DB" and [i,j] not in used_value:
            if (where[2][1][1] in table_json['tc_fast'] and natsql_idx_to_table2_idx(table_json['tc_fast'].index(where[2][1][1]),table_json) in sq.full_db_match[i][j]) or (type(where[2][1][1]) == int and natsql_idx_to_table2_idx(where[2][1][1],table_json) in sq.full_db_match[i][j]):
                re_num,used_value = add_one_value(pt,"DB",pts,sq,used_value,i,j)
                return True,re_num,used_value
        elif ((patterns == ["DB"] and pt == "DB") or (patterns == ["SDB"] and pt == "SDB") or (patterns == ["PDB"] and pt == "PDB") or (patterns == ["UDB"] and pt == "UDB") or (patterns == ["NN"] and pt == "NN")) and [i,j] not in used_value:
            re_num,used_value = add_one_value(pt,pt,pts,sq,used_value,i,j)
            return True,re_num,used_value
        elif patterns == ["tilt"] and pt == "tilt" and [i,j] not in used_value:
            return search_like(sq,used_value,i,j,sql_dict,where_idx)
        elif patterns == ["BOOL_NUM","NUM"] and pt in ["NUM"] and [i,j] not in used_value:
            re_num = sq.sub_sequence_toks[i][j]
            if (re_num.isdigit() and re_num in ["1","0"]) or re_num in ["'1'","'0'",'"1"','"0"']:
                used_value.append([i,j])
                re_num = str2num(re_num)
                return True,re_num,used_value
        elif patterns == ["DB_NUM","NUM"] and pt == "NUM" and [i,j] not in used_value:
            re_num = sq.sub_sequence_toks[i][j]
            re_num = str2num(re_num)
            used_value.append([i,j])
            return True,re_num,used_value
    return False,0,used_value

def analyse_num(sq,used_value,sql_dict,w_i,where,available_pattern,table_json,sq_sub_idx=-1):
    def set_2nd_between_value(num,sql_dict,w_i):
        if type(num) == int and num < sql_dict['where'][w_i][3]:
            sql_dict['where'][w_i][4] = sql_dict['where'][w_i][3]
            sql_dict['where'][w_i][3] = num
        else:
            sql_dict['where'][w_i][4] = num
        return sql_dict

    num_count = 0
    success = False
    for (i_t,type_), pts in zip(enumerate(sq.sub_sequence_type),sq.pattern_tok):
        if sq_sub_idx >= 0 and sq_sub_idx != i_t:
            continue
        there_is_available_pattern = False
        for ap in available_pattern:
            if ap in pts:
                there_is_available_pattern = True
                break
        if there_is_available_pattern:
            success,num,used_value = generate_where_values(i_t,pts,sq,used_value,available_pattern,where,table_json,sql_dict,w_i)
            if not success:
                continue
            else:
                if where[1] == 1 and num_count == 1:
                    sql_dict = set_2nd_between_value(num,sql_dict,w_i)
                else:
                    sql_dict['where'][w_i][3] = num
                    num_count = 1
                    if where[1] != 1:
                        break
                    success,num,used_value = generate_where_values(i_t,pts,sq,used_value,available_pattern,where,table_json,sql_dict,w_i)
                    if success:
                        sql_dict = set_2nd_between_value(num,sql_dict,w_i)
                        break
    return success,used_value,sql_dict


def fill_values(sql_dict, sq, table_json):    

    if not sq:
        return sql_dict
    
    used_value = []
    
    if sql_dict['limit']:
        for (i,type_), pts in zip(enumerate(sq.sub_sequence_type),sq.pattern_tok):
            if "NUM" in pts and ("GR_JJS" in pts or "SM_JJS" in pts or "top" in pts):
                num_idx,num = get_num_for_limit(pts,sq,sql_dict,i,table_json)
                if num and type(num) == int:
                    sql_dict['limit'] = num
                    used_value.append([i,num_idx])
        if sql_dict['limit'] == 1:
            num_in_select = False
            there_is_agg = False
            for (i,type_), pts in zip(enumerate(sq.sub_sequence_type),sq.pattern_tok):
                if type_ <= 1 and "NUM" in pts:
                    num_in_select = True
                if "GR_JJS" in pts or "SM_JJS" in pts:
                    there_is_agg  = True
            if there_is_agg and num_in_select:
                for (i,type_), pts in zip(enumerate(sq.sub_sequence_type),sq.pattern_tok):
                    if type_ <= 1 and "NUM" in pts:
                        num_idx,num = get_num_for_limit(pts,sq,sql_dict,i,table_json)
                        if num and type(num) == int:
                            sql_dict['limit'] = num
                            used_value.append([i,num_idx])
    
    for i,where in enumerate(sql_dict['where']):
        if type(where) != list or where[2][1][1] not in table_json['tc_fast']:
            continue
        elif where[1] in [1,3,4,5,6,2,7,14] and (where[1] in [1,3,4,5,6] or where[2][1][0] or table_json['column_types'][table_json['tc_fast'].index(where[2][1][1])] in ["number","year","time"]) and where[3] == '"terminal"':
            success,used_value,sql_dict = analyse_num(sq,used_value,sql_dict,i,where,["NUM","DATE","YEAR"],table_json)
            if not success:
                success,used_value,sql_dict = analyse_num(sq,used_value,sql_dict,i,where,["DB","SDB"],table_json)
            if not success and len(sql_dict['where']) == 1 and where[1] == 2:
                success,used_value,sql_dict = analyse_num(sq,used_value,sql_dict,i,where,["NOT"],table_json)
            if not success and i-2>=0 and type(sql_dict['where'][i-2]) == list and (sql_dict['where'][i-2][1] in [1,3,4,5,6] or sql_dict['where'][i-2][2][1][0] or table_json['column_types'][table_json['tc_fast'].index(sql_dict['where'][i-2][2][1][1])] in ["number","year","time"]) and type(sql_dict['where'][i-2][3]) != list and sql_dict['where'][i-2][3] != '"terminal"':
                sql_dict['where'][i][3] = sql_dict['where'][i-2][3]
                success = True
            if not success and i-4>=0 and type(sql_dict['where'][i-4]) == list and sql_dict['where'][i-4][1] == sql_dict['where'][i][1] and sql_dict['where'][i][2][1][1] == sql_dict['where'][i-4][2][1][1] and type(sql_dict['where'][i-4][3]) != list and sql_dict['where'][i-4][3] != '"terminal"':
                sql_dict['where'][i][3] = sql_dict['where'][i-4][3]
                success = True
            if not success:
                if table_json['data_samples'][table_json['tc_fast'].index(where[2][1][1])] and type(table_json['data_samples'][table_json['tc_fast'].index(where[2][1][1])][0]) != str:
                    sql_dict['where'][i][3] = table_json['data_samples'][table_json['tc_fast'].index(where[2][1][1])][0]
                else:
                    sql_dict['where'][i][3] = 1
        elif where[1] in [2,7] and  table_json['column_types'][table_json['tc_fast'].index(where[2][1][1])] in ["text"] and where[3] == '"terminal"':
            success,used_value,sql_dict = analyse_num(sq,used_value,sql_dict,i,where,["TEXT_DB","DB"],table_json)
            if not success and no_more_num_cond(sql_dict['where'],i,table_json):
                success,used_value,sql_dict = analyse_num(sq,used_value,sql_dict,i,where,["DB_NUM","NUM"],table_json)
            if not success:
                success = search_db(sq,table_json,where,used_value)
            if not success and i-4>=0 and type(sql_dict['where'][i-4]) == list and sql_dict['where'][i-4][1] == sql_dict['where'][i][1] and sql_dict['where'][i][2][1][1] == sql_dict['where'][i-4][2][1][1] and type(sql_dict['where'][i-4][3]) != list and sql_dict['where'][i-4][3] != '"terminal"':
                sql_dict['where'][i][3] = sql_dict['where'][i-4][3]
                success = True
    # BOOL
    for i,where in enumerate(sql_dict['where']):
        if type(where) != list or where[2][1][1] not in table_json['tc_fast']:
            continue
        elif where[1] in [2,7] and  table_json['column_types'][table_json['tc_fast'].index(where[2][1][1])] in ["boolean"] and where[3] == '"terminal"':
            success,used_value,sql_dict = analyse_num(sq,used_value,sql_dict,i,where,["DB"],table_json)
            if not success:
                success,used_value,sql_dict = analyse_num(sq,used_value,sql_dict,i,where,["PDB"],table_json)
            if not success:
                success,used_value,sql_dict = analyse_num(sq,used_value,sql_dict,i,where,["UDB"],table_json)
            if not success:
                success,used_value,sql_dict = analyse_num(sq,used_value,sql_dict,i,where,["SDB"],table_json)
            if not success:
                success,used_value,sql_dict = analyse_num(sq,used_value,sql_dict,i,where,["BOOL_NUM","NUM"],table_json)
            if not success:
                search_bcol(sq,table_json,where,sql_dict,i)
            if type(where[3]) == str and where[3][1:-1] not in table_json["data_samples"][table_json['tc_fast'].index(where[2][1][1])]:
                for v in table_json["data_samples"][table_json['tc_fast'].index(where[2][1][1])]:
                    if type(v) == str and (where[3][1:].lower().startswith(v.lower()) or v.lower().startswith(where[3][1:].lower())):
                        where[3] = "'"+v+"'"
            if i == 2 and len(sql_dict['where']) == 3 and type(sql_dict['where'][0]) == list and sql_dict['where'][0][2][1][1] == where[2][1][1] and where[3] == sql_dict['where'][0][3]:
                for v in table_json["data_samples"][table_json['tc_fast'].index(where[2][1][1])]:
                    if v != where[3][1:-1]:
                        if type(v) == str:
                            where[3] = "'"+v+"'"
                        else:
                            where[3] = v
                        break
            if type(where[3]) == str and where[3][1:-1] not in table_json["data_samples"][table_json['tc_fast'].index(where[2][1][1])] and table_json["data_samples"][table_json['tc_fast'].index(where[2][1][1])]:
                if where[3].startswith("'") and where[3].endswith("'") and not str_is_num(where[3][1:-1]) and table_json["data_samples"][table_json['tc_fast'].index(where[2][1][1])]:
                    success,used_value,sql_dict = analyse_num(sq,used_value,sql_dict,i,where,["BOOL_NUM","NUM"],table_json)
                    if not success:
                        where[3] = '"terminal"'
                        search_bcol(sq,table_json,where,sql_dict,i)

    for i,where in enumerate(sql_dict['where']):
        if type(where) != list or where[2][1][1] not in table_json['tc_fast']:
            continue
        elif where[1] in [9,13] and where[3] == '"terminal"':
            success,used_value,sql_dict = analyse_num(sq,used_value,sql_dict,i,where,["PDB"],table_json)
            if not success:
                success,used_value,sql_dict = analyse_num(sq,used_value,sql_dict,i,where,["UDB"],table_json)
            if not success:
                success,used_value,sql_dict = analyse_num(sq,used_value,sql_dict,i,where,["tilt"],table_json)
            if not success:
                success,used_value,sql_dict = analyse_num(sq,used_value,sql_dict,i,where,["SDB"],table_json)
            if not success:
                success,used_value,sql_dict = analyse_num(sq,used_value,sql_dict,i,where,["DB"],table_json)
            if not success:
                success,used_value,sql_dict = analyse_num(sq,used_value,sql_dict,i,where,["NUM","DATE","YEAR"],table_json)
            if not success:
                where[3] = "'%'"
            else:
                if table_json['column_types_checked'][table_json['tc_fast'].index(where[2][1][1])] in ["date","year","time","number"]:
                    if type(where[3]) == str and where[3].startswith("'"):
                        stemmer = MyStemmer()
                        value=stemmer.restem(where[3][1:-1])
                        if value.isdigit():
                            where[3] = "'"+value+"%'"
                        elif str2num(where[3][1:-1]):
                            where[3] = "'"+str2num(where[3][1:-1])+"%'"
                        elif "%" not in where[3]:
                            where[3] = where[3][0] + "%" + where[3][1:-1] + "%" + where[3][-1]
                    else:
                        where[3] =  "'" + str(where[3]) + "%'"

            if "%" not in where[3] and len(where[3]) > 2:
                if where[3][1:-1].isdigit() or " start with " in sq.question_or or " starting with " in sq.question_or or " started with " in sq.question_or or " starts with " in sq.question_or or " begin with " in sq.question_or or " beginning with " in sq.question_or or " began with " in sq.question_or or " begins with " in sq.question_or or " start from " in sq.question_or or " starts from " in sq.question_or or " starting from " in sq.question_or or " started from " in sq.question_or or " begin from " in sq.question_or or " begins from " in sq.question_or or " began from " in sq.question_or or " beginning from " in sq.question_or:
                    where[3] = where[3][0:-1] + "%" + where[3][-1]
                elif " end with " in sq.question_or or " ending with " in sq.question_or or " ended with " in sq.question_or or " ends with " in sq.question_or or " end by " in sq.question_or or " ending by " in sq.question_or or " ended by " in sq.question_or or " ends by " in sq.question_or:
                    where[3] =  where[3][0] + "%" + where[3][1:]
                else:
                    where[3] = where[3][0] + "%" + where[3][1:-1] + "%" + where[3][-1]

    return sql_dict


def IUE2Where(sql_dict,table_json):
    def search_where_col(tbl,take_tbl,where_col,table_json,w_idx,sql_dict):
        tmp_take_tbl = [False,False]
        new_where_cond = [False,False]
        for i,t in enumerate(tbl):
            if t:
                t = t.lower()
                # search same original name
                col = t + "." + where_col.split(".")[1]
                for tc in  table_json['table_column_names_original']:
                    if tc[1].lower() == col:
                        tmp_take_tbl[i] = True
                        new_where_cond[i] = copy.deepcopy(sql_dict["where"][w_idx])
                        new_where_cond[i][2][1][1] = col
                        break
                if not tmp_take_tbl[i]:
                    # search same annotation name
                    where_col_idx = 0
                    for j,tc in  enumerate(table_json['table_column_names_original']):
                        if tc[1].lower() == where_col:
                            where_col_idx = j
                            break
                    t_idx = -1
                    for j,tor in enumerate(table_json['table_names_original']):
                        if tor.lower() == t:
                            t_idx = j
                            break
                    for col in table_json["column_names"]:
                        if col[0] == t_idx and col[1].lower() == table_json["column_names"][where_col_idx][1].lower():
                            tmp_take_tbl[i] = True
                            new_where_cond[i] = copy.deepcopy(sql_dict["where"][w_idx])
                            new_where_cond[i][2][1][1] = table_json['table_column_names_original'][j][1]
                            break
        if w_idx == 0 and tmp_take_tbl[0]:
            take_tbl[0] = True
        elif not tmp_take_tbl[0]:
            take_tbl[0] = False
        if w_idx == 0 and tmp_take_tbl[1]:
            take_tbl[1] = True
        elif not tmp_take_tbl[1]:
            take_tbl[1] = False
        return take_tbl,new_where_cond

    tbl = ["",""]
    take_tbl = [False,False]
    iue_where = [None,None]
    new_where = [[],[]]
    if sql_dict['union']:
        tbl[0] = sql_dict['union']['select'][1][0][1][1][1].split(".")[0]
    if sql_dict['intersect']:
        tbl[1] = sql_dict['intersect']['select'][1][0][1][1][1].split(".")[0]

    for w,where in enumerate(sql_dict['where']):
        if type(where) == list:
            where_col = where[2][1][1]
            if where_col != "@.@":
                take_tbl,new_where_cond = search_where_col(tbl,take_tbl,where_col,table_json,w,sql_dict)
            elif type(where[3]) == list and where[3][1] == "@.@":
                take_tbl,new_where_cond = search_where_col(tbl,take_tbl,where[3][1],table_json,w,sql_dict)
            if take_tbl[0]:
                new_where[0].append(new_where_cond[0])
            if take_tbl[1]:
                new_where[1].append(new_where_cond[1])
        else:
            if take_tbl[0]:
                new_where[0].append(where)
            if take_tbl[1]:
                new_where[1].append(where)


    if (not sql_dict['where']) or (len(sql_dict['where']) == 1 and sql_dict['where'][0][1] == 15):
        tbl = ["",""]
        take_tbl = [False,False]
        if sql_dict['union']:
            tbl[0] = sql_dict['union']['select'][1][0][1][1][1].split(".")[0]
            take_tbl[0] = True
        if sql_dict['intersect']:
            tbl[1] = sql_dict['intersect']['select'][1][0][1][1][1].split(".")[0]
            take_tbl[1] = True

    
    if take_tbl[0]:
        iue_where[0] = ["union_",[False, 10, [0, [0, '@.@', False], None], [0, sql_dict['union']['select'][1][0][1][1][1], False], None]]
        if len(sql_dict['where']) == 1 and sql_dict['where'][0][1] == 15:
            iue_where[0] = ["union_",[False, 15, [0, [0, '@.@', False], None], [0, sql_dict['union']['select'][1][0][1][1][1], False], None]]
        elif sql_dict['where']:
            iue_where[0]=["union_"]
            iue_where[0].extend(new_where[0])
    if take_tbl[1]:
        iue_where[1] = ["intersect_",[False, 10, [0, [0, '@.@', False], None], [0, sql_dict['intersect']['select'][1][0][1][1][1], False], None]]
        if len(sql_dict['where']) == 1 and sql_dict['where'][0][1] == 15:
            iue_where[1] = ["intersect_",[False, 15, [0, [0, '@.@', False], None], [0, sql_dict['intersect']['select'][1][0][1][1][1], False], None]]
        elif sql_dict['where']:
            iue_where[1]=["intersect_"]
            iue_where[1].extend(new_where[0])
    if iue_where[0]:
        sql_dict['where'].extend(iue_where[0])
    if iue_where[1]:
        sql_dict['where'].extend(iue_where[1])


def iue2subquery(from_table_net,select_table_list,table_json,groupby_list,where_left_col_list,sql_dict):
    t_id_list = []
    for t in select_table_list:
        t_id_list.append(table_json['table_orig_low'].index(t))
    if (sql_dict['where'][1] in ["except_","intersect_","union_"]):
        pass
    elif len(sql_dict['where']) == 5 and sql_dict['where'][3] in ["except_","intersect_","union_"] and type(sql_dict['where'][0]) == list and sql_dict['where'][0][2][1][1].lower() in table_json['tc_fast']:
        t_id_list.append(table_json['column_names'][table_json['tc_fast'].index(sql_dict['where'][0][2][1][1].lower())][0])
    if len(sql_dict['where']) == 5 and len(set(t_id_list)) == 1 and sql_dict['where'][3] in ["except_","intersect_","union_"]:
        insert_idx = 2
    else:
        insert_idx = 0
    for fks in from_table_net[0]:
        if table_json['column_names'][fks[0]][0]  in t_id_list and table_json['column_names'][fks[1]][0] not in t_id_list :
            sql_dict['where'].insert(insert_idx,"and")
            sql_dict['where'].insert(insert_idx,[False,8,[0,[0,"@.@",False],None],[0,table_json['tc_fast'][fks[1]],None],None])
            return True,sql_dict
        if table_json['column_names'][fks[1]][0]  in t_id_list and table_json['column_names'][fks[0]][0] not in t_id_list :
            
            sql_dict['where'].insert(insert_idx,"and")
            sql_dict['where'].insert(insert_idx,[False,8,[0,[0,"@.@",False],None],[0,table_json['tc_fast'][fks[0]],None],None])
            return True,sql_dict
    if len(sql_dict['select'][1]) == 1 and (sql_dict['select'][1][0][0] or sql_dict['select'][1][0][1][1][0]) and len(sql_dict['where']) in [3,5] and (sql_dict['where'][1] in ["except_","intersect_","union_"] or (len(sql_dict['where']) == 5 and sql_dict['where'][3] in ["except_","intersect_","union_"])):
        sql_dict['where'].insert(insert_idx,"and")
        sql_dict['where'].insert(insert_idx,[False,8,[0,[0,"@.@",False],None],[0,sql_dict['where'][insert_idx+1][2][1][1].split(".")[0]+".*",None],None])
        return True,sql_dict

    return False,sql_dict



def join2subquery(from_table_net,select_table_list,table_json,groupby_list,sql_dict):
    t_id_list = []
    for t in select_table_list:
        t_id_list.append(table_json['table_orig_low'].index(t))
    insert_idx = 0
    for i, w in enumerate(sql_dict["where"]):
        if type(w) == list and w[2][1][1].lower() in table_json['tc_fast'] and table_json['column_names'][table_json['tc_fast'].index(w[2][1][1].lower())][0] in t_id_list:
            insert_idx += 2
        elif type(w) == list:
            break

    group_col_idx = -1
    if len(groupby_list) == 1:
        group_col_idx = table_json['tc_fast'].index(groupby_list[0].lower())
    
    if len(t_id_list) >= 2:
        bridge_table = dict()
        for fks in from_table_net[0]:
            if table_json['column_names'][fks[0]][0] not in bridge_table:
                bridge_table[table_json['column_names'][fks[0]][0]] = set([fks[0]])
            else:
                bridge_table[table_json['column_names'][fks[0]][0]].add(fks[0])
            if table_json['column_names'][fks[1]][0] not in bridge_table:
                bridge_table[table_json['column_names'][fks[1]][0]] = set([fks[1]])
            else:
                bridge_table[table_json['column_names'][fks[1]][0]].add(fks[1])
        bridge_table = [ k  for k in bridge_table if len(bridge_table[k]) > 1]
    else:
        bridge_table = []

    for fks in from_table_net[0]:
        if fks[0] in table_json['primary_keys'] and fks[1] not in table_json['primary_keys'] and table_json['column_names'][fks[0]][0] in t_id_list and fks[1] not in table_json['unique_fk'] and table_json['column_names'][fks[1]][0] not in t_id_list and table_json['column_names'][fks[1]][0] not in bridge_table and group_col_idx not in fks:
            table_idx = table_json['column_names'][fks[1]][0]
            sql_dict['where'].insert(insert_idx,"and")
            sql_dict['where'].insert(insert_idx,[False,8,[0,[0,"@.@",False],None],[0,table_json['tc_fast'][fks[1]],None],None])
            return True,sql_dict
        if fks[1] in table_json['primary_keys'] and fks[0] not in table_json['primary_keys'] and table_json['column_names'][fks[1]][0] in t_id_list and fks[0] not in table_json['unique_fk'] and table_json['column_names'][fks[0]][0] not in t_id_list and table_json['column_names'][fks[0]][0] not in bridge_table and group_col_idx not in fks:
            table_idx = table_json['column_names'][fks[0]][0]
            sql_dict['where'].insert(insert_idx,"and")
            sql_dict['where'].insert(insert_idx,[False,8,[0,[0,"@.@",False],None],[0,table_json['tc_fast'][fks[0]],None],None])
            return True,sql_dict
    return False,sql_dict


def groupby2subquery(from_table_net,select_table_list,table_json,groupby_list,sql_dict):
    group_col_idx = table_json['tc_fast'].index(groupby_list[0].lower())
    t_id_list = []
    for t in select_table_list:
        t_id_list.append(table_json['table_orig_low'].index(t))
    for fks in from_table_net[0]:
        if group_col_idx in fks:
            return False,sql_dict
    if table_json['column_names'][group_col_idx][0] not in t_id_list:
        pass
    elif group_col_idx in table_json['primary_keys'] or (len(sql_dict['select'][1]) == 1 and sql_dict['select'][1][0][1][1][1].lower() == groupby_list[0].lower()):
        return False,sql_dict
    else:
        pass
    insert_idx = 0
    for i, w in enumerate(sql_dict["where"]):
        if type(w) == list and not w[2][1][0]:
            insert_idx += 2
        elif type(w) == list:
            break
    sql_dict['where'].insert(insert_idx,"and")
    sql_dict['where'].insert(insert_idx,[False,8,[0,[0,"@.@",False],None],[0,table_json['tc_fast'][group_col_idx],None],None]) 
    sql_dict['groupBy'] = []
    return True,sql_dict


def simplify_join(from_table_net,select_table_list,table_json,groupby_list,sql_dict):
    if sql_dict['where'] and type(sql_dict['where'][0]) == list and sql_dict['where'][0][1] == 12 and sql_dict['where'][0][2][1][1] == sql_dict['where'][0][3][1]:
        where_col_idx = table_json['tc_fast'].index(sql_dict['where'][0][2][1][1].lower())
        need_simplify = False
        other_col = 0
        for fks in from_table_net[0]:
            if where_col_idx in fks:
                need_simplify = True
                other_col = fks[1] if where_col_idx == fks[0] else fks[0]
        if need_simplify:
            sql_dict['where'][0][2][1][1] = table_json['tc_fast'][other_col]
            return True,sql_dict
    return False,sql_dict        
    


def search_all_join_on(sql_dict, table_json, args, join_on_label=None, sq=None, first_round=True):
    """
    modified from NatSQL V1.0 inference code.
    """
    if not sql_dict:
        return 0," ",[],sql_dict
    sql_dict_orig = copy.deepcopy(sql_dict)
    all_from = []
    global globe_join_on_label_count
    globe_join_on_label_count = 0
    
    top_groupby_list = []
    groupby_list = []
    groupby_top = ""
    add_top_group = True
    re_sql = "select distinct " if sql_dict['select'][0] else "select "
    orderby_sql,table_list,agg_in_order = ("",[],False)
    agg_in_select = False
    # Get table info from select column
    for column in sql_dict['select'][1]:
        table = column[1][1][1].split('.')[0].lower()
        if not table in table_list:
            table_list.append(table)
        select_unit = select_unit_back(column)
        if not (column[0] or column[1][1][0]):
            if select_unit != '*':
                top_groupby_list.append(column[1][1][1].lower() if column[1][0] else select_unit)  
        else:
            agg_in_select = True
        re_sql += select_unit + ' , '
    re_sql = re_sql[:-3]
    top_select_table_list = copy.deepcopy(table_list)
    
    if sql_dict['union'] or sql_dict['intersect']:
        IUE2Where(sql_dict,table_json)
    sql_dict['where'] = intersect_where_order(sql_dict['where'],top_select_table_list)
    if first_round:
        sql_dict['where'] = except_inference(sql_dict['where'],sq)
        sql_dict['where'] = intersect_inference(sql_dict['where'],sq)
        sql_dict['where'] = union_inference(sql_dict['where'],sq,table_json,top_select_table_list)
    
    try:
        if args.fill_value:
            sql_dict = fill_values(sql_dict, sq, table_json)
    except:
        pass
    if sql_dict['from']['table_units'] and args.use_from_info:
        for t in sql_dict['from']['table_units']:
            if t[1].lower() not in table_list:
                table_list.append(t[1])

    # Add table info to select column
    break_idx,table_list,next_sql,sql_where,sql_having,orderby_sql_,next_table_list,top_left_col_list = get_where_column(sql_dict, table_list, 0, SQL_TOP, table_json, args)
    if (" * not in " in sql_where or " * in " in sql_where) and len(sql_dict['where']) - break_idx >= 3 and sql_dict['where'][break_idx][1] in [12,8] and sql_dict['where'][break_idx+2][2][1][1].split(".")[0] != sql_dict['where'][break_idx][3][1].split(".")[0]:
        next_table_list = [sql_dict['where'][break_idx][3][1].split(".")[0]]
        sql_dict['where'][break_idx][2][1][1] = "@.@"
        if sql_dict['where'][break_idx+2][2][1][1].split(".")[0] != "@":
            sql_dict['where'][break_idx][3][1] = sql_dict['where'][break_idx+2][2][1][1].split(".")[0]+".*"
        else:
            sql_dict['where'][break_idx][3][1] = sql_dict['where'][break_idx+2][3][1].split(".")[0]+".*"
        break_idx,_,next_sql,sql_where,sql_having,orderby_sql_,__,top_left_col_list = get_where_column(sql_dict, table_list, 0, SQL_TOP, table_json, args)


    if break_idx < 0 or next_sql == SQL_TOP or (sql_dict['orderBy'] and not sql_dict['limit']):
        orderby_sql,table_list_order,agg_in_order = create_order_by(sql_dict['orderBy'],sql_dict['limit'])
        for order_t in table_list_order:
            if order_t.lower() not in table_list:
                table_list.append(order_t.lower())
    
    orderby_sql += orderby_sql_
    from_table_net,table_fk_list = get_table_network(table_json, table_list, join_on_label, sq=sq, sql_dict=sql_dict, group_list=top_left_col_list+top_groupby_list if top_groupby_list else [] )
    from_table_netss,_ = get_table_network(table_json, table_list, join_on_label, False, group_list=top_left_col_list+top_groupby_list if top_groupby_list else [])
    all_from.append(from_table_netss)

    if sql_dict['groupBy']: #V1.1:
        groupby_list = [col_unit_back(gBy) for gBy in sql_dict['groupBy']]
        groupby_top = " group by " + ", ".join(groupby_list)
        for gBy in sql_dict['groupBy']:
            table_list.append(gBy[1].split(".")[0])
        table_list = list(set(table_list))
        if break_idx == SQL_TOP:
            add_top_group = False
        if (len(top_groupby_list) != len(sql_dict['select'][1]) and top_groupby_list) or sql_having.strip() != '' or (agg_in_order and top_groupby_list) or orderby_sql_.strip():
            add_top_group = True
    elif not args.not_infer_group and (len(top_groupby_list) != len(sql_dict['select'][1]) and top_groupby_list) or sql_having.strip() != '' or (agg_in_order and top_groupby_list) or orderby_sql_.strip():
        table_list = list(set(table_list))
        if args.group_for_exact_match and (len(top_groupby_list) > 1 or len(table_list) > 1 or len(top_groupby_list) == 0):
            agg_tables = get_agg_tables(sql_dict,table_json)
            groupby_list = infer_group_for_exact_match(top_groupby_list,table_json,table_list,len(sql_dict['select'][1]),sq,agg_tables,from_table_net)
            groupby_list = group_back_because_eva_bug_in_spider(groupby_list,table_json,agg_tables,top_groupby_list,from_table_net)
            groupby_top = " group by " + ",".join(groupby_list)
        else:
            groupby_top = " group by " + ",".join(top_groupby_list)
    else:
        table_list = list(set(table_list))
    
    if top_groupby_list and not groupby_list:
        groupby_list = top_groupby_list
    if args.iue2subquery and agg_in_select and ((len(sql_dict['where']) in [3,5] and sql_dict['where'][1] in ["except_","intersect_","union_"]) or (len(sql_dict['where']) == 5 and sql_dict['where'][3] in ["except_","intersect_","union_"])):
        success,sql_dict = iue2subquery(from_table_net,top_select_table_list,table_json,groupby_list,top_left_col_list,sql_dict)
        if success:
            return 1,sql_dict,None,None
    if args.join2subquery and agg_in_select and not sql_dict['select'][0] and not sql_dict['limit'] and not (sql_dict['where'] and type(sql_dict['where'][0])==list and sql_dict['where'][0][1] in [12,8]):
        success,sql_dict_orig = join2subquery(from_table_net,top_select_table_list,table_json,groupby_list,sql_dict_orig)
        if success:
            return 2,sql_dict_orig,None,None
    if args.groupby2subquery and groupby_top and len(groupby_list) == 1 and not agg_in_select:
        success,sql_dict_orig = groupby2subquery(from_table_net,top_select_table_list,table_json,groupby_list,sql_dict)
        if success:
            return 3,sql_dict_orig,None,None
    success,sql_dict = simplify_join(from_table_net,top_select_table_list,table_json,groupby_list,sql_dict)
    if success:
        return 4,sql_dict,None,None

    top_sql_list = [re_sql]
    re_sql += create_from_table(from_table_net,table_json['table_names_original'], table_json['table_column_names_original'], table_fk_list, table_list=table_list)
    if add_top_group:
        top_sql_list.append(re_sql + sql_where + groupby_top + sql_having)
    else:
        top_sql_list.append(re_sql + sql_where + sql_having)
        
    if sql_dict['where']:
        sub_sql_list = []
        sub_sql_select_table_list = []
        while next_sql:
            top_sql_to_sub = False
            previous_table_list = table_list
            table_list = next_table_list#V1.2
            if next_sql == SQL_TOP:
                if type(sql_dict['where'][0]) == str and sql_dict['where'][1][2][1][1] == "@.@" and sql_dict['where'][1][1] == 10:# and  ".*" in sql_dict['where'][1][3][1]:
                    # V1.4 NatSQL extension: where IUE @.@ is table.* :
                    if not (len(sql_dict['where']) > 3 and sql_dict['where'][2] == "and"):
                        sub_sql, table_list = infer_IUE_select_col(sql_dict,table_json,break_idx,top_sql_list,top_select_table_list)
                    else:
                        sub_sql, table_list = fk_replace_IUE(sql_dict,table_json,break_idx, top_sql_list,top_select_table_list)
                    break_idx += 2
                else:
                    if len(top_sql_list) == 2 and sql_dict['where'][break_idx] in ["intersect_","union_","except_"] and break_idx > 1 and sub_sql_list and sub_sql_select_table_list and sql_dict['where'][break_idx-1][2][1] == sql_dict['where'][break_idx+1][2][1]  and top_sql_list[1].strip().endswith(")"):
                        # IUE in sub query
                        sub_sql, table_list = fk_replace_IUE(sql_dict,table_json,break_idx, sub_sql_list,sub_sql_select_table_list)
                        top_sql_to_sub = True
                    elif len(top_sql_list) == 2 and sql_dict['where'][break_idx] in ["intersect_"] and break_idx > 1 and sql_dict['where'][break_idx-1][2] == sql_dict['where'][break_idx+1][2] and (break_idx == 1 or (break_idx == 3 and type(sql_dict['where'][0]) == list and sql_dict['where'][0][1] == 15)):
                        sub_sql, table_list = fk_replace_IUE(sql_dict,table_json,break_idx, top_sql_list,previous_table_list)
                    else:
                        sub_sql, table_list = fk_replace_IUE(sql_dict,table_json,break_idx, top_sql_list,top_select_table_list)

                start_new_top_sql = True
            else:
                select_column = col_unit_back(sql_dict['where'][break_idx][3])
                sub_sql = "select " + select_column
                sub_sql_list = ["select " + select_column]
                if '.' not in select_column:
                    sub_sql_select_table_list = [sql_dict['where'][break_idx][3][1].split(".")[0].lower()]
                else:
                    sub_sql_select_table_list = [sql_dict['where'][break_idx][3][1].split(".")[0].lower()]#[select_column.split(".")[0].lower()]
                if sql_dict['where'][break_idx][3][1].split('.')[0].lower() not in table_list:
                    table_list.append(sql_dict['where'][break_idx][3][1].split('.')[0].lower())
                start_new_top_sql = False

            previous_break_idx = break_idx
            break_idx,table_list,next_sql,sql_where,sql_having,orderby_sql_,next_table_list,__ = get_where_column(sql_dict, table_list, break_idx + 1, next_sql, table_json, args)
            if args.orderby_to_subquery and not orderby_sql_ and not (previous_break_idx > 1 and sql_dict["where"][previous_break_idx-1] == "sub"):
                orderby_sql_, table_list = orderby_to_subquery(sql_dict,table_list) #v1.1

            table_list = list(set(table_list))
            if sub_sql.startswith(" except "):
                from_table_net,table_fk_list = get_table_network(table_json, table_list, join_on_label)
            else:
                from_table_net,table_fk_list = get_table_network(table_json, table_list, join_on_label, sq=sq, sql_dict=sql_dict)
            from_table_netss,_ = get_table_network(table_json, table_list, join_on_label, False)
            all_from.append(from_table_netss)

            if sub_sql.startswith("select ") and select_column.count(".") == 1: # sub query
                sub_sql_table = select_column.split(".")[0]
                if "(" in sub_sql_table:
                    sub_sql_table = sub_sql_table.split("(")[1]
                sub_sql += create_from_table(from_table_net,table_json['table_names_original'], table_json['table_column_names_original'],table_fk_list,sub_sql_table, table_list=table_list)
            else:
                sub_sql += create_from_table(from_table_net,table_json['table_names_original'], table_json['table_column_names_original'],table_fk_list, table_list=table_list)
            
            sub_sql += sql_where
            

            if not start_new_top_sql:
                if (sql_having.strip() and select_column) or ( ("max(" in orderby_sql_ or "min(" in orderby_sql_ or "count(" in orderby_sql_ or "sum(" in orderby_sql_ or "avg(" in orderby_sql_) and select_column):#v1.0
                    sub_sql += " group by " + select_column
            else:
                if (sql_having.strip() != '' and groupby_list)  or (orderby_sql_.strip() and groupby_list):
                    if groupby_list and groupby_list[0].split(".")[0] not in table_list:
                        groupby_sub = extract_select_columns(sub_sql)
                        if groupby_sub:
                            sub_sql += " group by " + groupby_sub
                    elif groupby_top.strip():
                        sub_sql += groupby_top
                    else:
                        groupby_list = infer_group_for_exact_match(groupby_list,table_json,table_list,len(sql_dict['select'][1]))
                        sub_sql += " group by " + ",".join(groupby_list)

            sub_sql += sql_having + orderby_sql_

            if top_sql_to_sub and len(top_sql_list) == 2 and (top_sql_list[1].strip().endswith(")")):
                top_sql_list[-1] = top_sql_list[-1].strip()[:-1] + sub_sql + " ) "
            elif start_new_top_sql:
                top_sql_list.append(sub_sql)
            else:
                top_sql_list[len(top_sql_list)-1] = top_sql_list[len(top_sql_list)-1].replace('@@@',sub_sql,1)
    
    re_sql = ""
    for idx, sql in enumerate(top_sql_list):
        if idx > 0:
            re_sql += sql

    re_sql += orderby_sql
    
    return 0,re_sql,all_from,sql_dict
