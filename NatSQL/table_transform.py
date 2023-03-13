import os, copy, argparse, json, pickle
from natsql2sql.preprocess.TokenString import get_spacy_tokenizer
from natsql2sql.preprocess.match import STOP_WORDS
from natsql2sql.preprocess.Schema_Token import Schema_Token
from natsql2sql.preprocess.stemmer import MyStemmer
from natsql2sql.preprocess.db_match import DBEngine


AGG_OPS = ('none', 'max', 'min', 'count', 'sum', 'avg')
AGG_LANGUAGE = ('none', 'maximum', 'minimum', 'number of', 'sum of', 'average')

def construct_hyper_param():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_file', default='', type=str)
    parser.add_argument("--out_file", default='', type=str, help="output table.json")
    parser.add_argument("--star_type", default='others', type=str,
                        help="what type of * in column type?")
    parser.add_argument('--table_transform', action='store_true', default=False)

    parser.add_argument('--seperate_col_name', action='store_true', default=False)

    parser.add_argument('--modify_column_names', action='store_true', default=False)
    
    parser.add_argument('--use_table_name_for_star_in_col_name', action='store_true', default=False)

    parser.add_argument('--force_modify_star_column_names', action='store_true', default=False)

    parser.add_argument('--remove_star_from_col', action='store_true', default=False)#Only for training. If True, put star to table name.
    
    parser.add_argument('--add_alpha_to_table', action='store_true', default=False)#For syntaxSQL. Here is True, modify_column_names is True
    
    parser.add_argument('--correct_col_type', action='store_true', default=False)
    parser.add_argument('--use_extra_col_types', action='store_true', default=False)

    parser.add_argument('--remove_start_table', action='store_true', default=False)

    parser.add_argument('--recover_previous_column_content', action='store_true', default=False)

    parser.add_argument('--analyse_same_column', action='store_true', default=False)

    parser.add_argument('--add_star_on_first_col', action='store_true', default=False)

    parser.add_argument('--add_debug_col', action='store_true', default=False)

    parser.add_argument('--keepOriginal', action='store_true', default=False) # keep the original data as same as tables.json

    parser.add_argument('--correct_primary_keys', action='store_true', default=False) 

    parser.add_argument('--db_path', default='./database', type=str)
    
    args = parser.parse_args()
    return args



def reversed_link_back_col(col_id, table_json):
    for lb in range(col_id, len(table_json['link_back'])):
        if table_json['link_back'][lb][1] == col_id:
            return table_json['link_back'][lb][0]
    return 0

def table_transform(table, args, schema):
    new_table = copy.deepcopy(table)
    table_num = len(table['table_names'])
    
    # 1. Process column name and type:
    last_table_index,insert_index,item_index = [-1,-1,-1]
    insert_index_list = []
    new_table['link_back'] = []
    new_table['table_column_names_original'] = []

    table_name_list = [item[1] for item in table['column_names']]

    # The order of column_names may be different from column_names_original
    for item in table['column_names']:
        insert_index += 1
        item_index += 1
        if item[0] == -1:
            if args.add_alpha_to_table:
                new_table['column_names'][insert_index] = [-1,"@"]
                new_table['column_names_original'][insert_index] = [-1,"@"]
                new_table['table_column_names_original'].append([-1, "@"])
                new_table['link_back'].append([0,0])
                if 'old_column_names' in new_table:
                    new_table['old_column_names'][insert_index] = [-1,"@"]
            else:
                new_table['column_names'].pop(insert_index)
                new_table['column_names_original'].pop(insert_index)
                new_table['column_types'].pop(insert_index)
                if 'column_types_checked' in new_table:
                    new_table['column_types_checked'].pop(insert_index)
                    new_table['data_samples'].pop(insert_index)
                if 'old_column_names' in new_table:
                    new_table['old_column_names'].pop(insert_index)                 
                insert_index -= 1
            continue
        
        if last_table_index != item[0] and not args.remove_star_from_col:
            if args.modify_column_names:
                new_table['column_names'].insert(insert_index, [item[0], table['table_names'][item[0]].replace(" | "," * | ") + ' *'])
            else:
                if args.force_modify_star_column_names:
                    new_table['column_names'].insert(insert_index, [item[0], table['table_names'][item[0]].replace(" | "," * | ") + ' *'])
                else:
                    if args.use_table_name_for_star_in_col_name:
                        new_table['column_names'].insert(insert_index, [item[0], table['table_names'][item[0]]])
                    else:
                        new_table['column_names'].insert(insert_index, [item[0], '*'])
            new_table['column_names_original'].insert(insert_index, [item[0],'*'])
            new_table['column_types'].insert(insert_index, args.star_type)
            if 'column_types_checked' in new_table:
                new_table['column_types_checked'].insert(insert_index, "None")
                new_table['data_samples'].insert(insert_index, [])
            insert_index_list.append(insert_index)
            new_table['link_back'].append([insert_index, 0])
            new_table['table_column_names_original'].append([item[0], table['table_names_original'][item[0]]+'.*'])
            insert_index += 1

        last_table_index = item[0]
        if new_table['link_back']:
            insert_index_schema = new_table['link_back'][-1][1] + 1
        elif args.remove_star_from_col:
            insert_index_schema = insert_index + 1
        else:
            insert_index_schema = insert_index
        
        if args.modify_column_names and schema.column_tokens_lemma_str.count(schema.column_tokens_lemma_str[insert_index_schema]) > 1:
            for (tb_idx, last_table_string),last_table_lemma in zip(enumerate(table['table_names']),schema.table_tokens_lemma_str):
                last_table_string = last_table_string.split(" | ")
                last_table_lemma = last_table_lemma.split(" | ")
                if not new_table['column_names'][insert_index][1].startswith(last_table_string[0]+" ") and new_table['column_names'][insert_index][1] != last_table_string[0] \
                    and not schema.column_tokens_lemma_str[insert_index_schema].startswith(last_table_lemma[0]+" ") and schema.column_tokens_lemma_str[insert_index_schema] != last_table_lemma[0] \
                    and ( len(last_table_string)==1 or (not schema.column_tokens_lemma_str[insert_index_schema].startswith(last_table_lemma[-1]+" ") and schema.column_tokens_lemma_str[insert_index_schema] != last_table_lemma[-1])  ) \
                    and " " + schema.column_tokens_lemma_str[insert_index_schema] not in last_table_lemma[0] and  schema.column_tokens_lemma_str[insert_index_schema] + " " not in last_table_lemma[0]:
                    pass
                else:
                    for i2,col2 in enumerate(new_table['column_names']):
                        if new_table['column_names'][i2][1] == new_table['column_names'][insert_index][1] and new_table['column_names'][i2][0] == tb_idx:
                            last_table_string = None
                            break
                    if not last_table_string:
                        break
            if last_table_string:
                last_table_string = table['table_names'][last_table_index].split(" | ")
                if " | " not in new_table['column_names'][insert_index][1] and len(last_table_string) > 1:
                    new_table['column_names'][insert_index][1] = " | ".join([lts + " " + new_table['column_names'][insert_index][1] for lts in last_table_string])
                else:
                    cols = new_table['column_names'][insert_index][1].split(" | ")
                    for i,cc in enumerate(cols):
                        cols[i] = last_table_string[0] + " " + cc
                        new_table['column_names'][insert_index][1] = " | ".join(cols)
        new_table['link_back'].append([insert_index, item_index])
        new_table['table_column_names_original'].append([item[0], table['table_names_original'][item[0]]+"."+table['column_names_original'][item_index][1] ] )

    if not args.remove_star_from_col:
        add = 0 if args.add_alpha_to_table else 1
        assert len(new_table['column_names'])+add-table_num ==  len(table['column_names'])
    assert len(new_table['column_names']) ==  len(new_table['column_names_original'])


    # 2. Process keys:
    item_index = 0
    for pkey in new_table['primary_keys']:
        for nidx,oidx in new_table['link_back']:
            if oidx == pkey:
                new_table['primary_keys'][item_index] = nidx
                item_index += 1
                break
    if 'original_primary_keys' in new_table:
        item_index = 0
        for pkey in new_table['original_primary_keys']:
            for nidx,oidx in new_table['link_back']:
                if oidx == pkey:
                    new_table['original_primary_keys'][item_index] = nidx
                    item_index += 1
                    break
    break_token = 0
    item_index = 0
    for fkey_1,fkey_2 in new_table['foreign_keys']:
        for nidx,oidx in new_table['link_back']:
            if oidx == fkey_1:
                new_table['foreign_keys'][item_index][0] = nidx
                break_token += 1
            if oidx == fkey_2:
                new_table['foreign_keys'][item_index][1] = nidx
                break_token += 1
            if break_token == 2:
                break
        break_token = 0
        item_index += 1

    return new_table


def get_next_table(table_idx,table_range,table_fk,table_col):
    return_table_list = []
    return_fk_list = []
    col_range = table_range[table_idx]
    for fk in table_fk:
        if fk[0]>= col_range[0] and fk[0]<= col_range[1]:  # Check the left fk belong to the table of table_idx
            return_table_list.append(table_col[fk[1]][0])
            return_fk_list.append(fk)
    return return_table_list,return_fk_list


def build_table_network(table):
    """
    net work example:
    [ 
        [[], [1]],[[], [2]],[[], [3]],
        [[[fk1,fk2]],[1,2]],[[[fk1,fk3]],[1,3]],[[[fk2,fk3]],[2,3]],
        [ [[fk1,fk2],[fk2,fk3]],[1,2,3] ]
    ]
    """

    table['network'] = []
    for idx, _ in enumerate(table['table_names_original']):
        table['network'].append([[],[idx]])
    
    table_net_idx = 0
    back_count = 1
    table_all = []
    max_table_net_len = 2

    while True:
        one_net_table_list = table['network'][table_net_idx][1]
        index_table_in_table_list = len(one_net_table_list)-back_count
        fk_of_one_net_table_list = None
        for jj in range(2):
            if  index_table_in_table_list >= 0:
                table_fk_left = one_net_table_list[index_table_in_table_list]
                next_tables,fk_next_tables = get_next_table(table_fk_left\
                    ,table['index_range'],table['foreign_keys'],table['column_names_original'])
                
                if next_tables:
                    for nt,fk in zip(next_tables,fk_next_tables):
                        if nt not in table['network'][table_net_idx][1]:
                            if jj == 1 and fk_of_one_net_table_list:
                                new = copy.deepcopy([fk_of_one_net_table_list,one_net_table_list])
                            else:
                                new = copy.deepcopy(table['network'][table_net_idx])
                            new[0].append(fk)
                            new[1].append(nt)
                            to_all_table = copy.deepcopy(new[1])
                            to_all_table.sort()
                            if to_all_table not in table_all:
                                table_all.append(to_all_table)
                                table['network'].append(new)
                                if len(new[1]) > max_table_net_len:
                                    max_table_net_len = len(new[1])
            
            if len(one_net_table_list) >= 2 and index_table_in_table_list == len(one_net_table_list) - 1:
                one_net_table_list = [r for r in reversed(one_net_table_list)]
                fk_of_one_net_table_list = [r for r in reversed(table['network'][table_net_idx][0])]
            else:
                break
                                
        table_net_idx += 1
        if table_net_idx >= len(table['network']):
            if max_table_net_len == back_count:
                break
            back_count += 1
            table_net_idx = len(table['table_names_original'])
            if table_net_idx >= len(table['network']):
                table_net_idx = len(table['network']) - 1 # it should break here

    return table



def expand_foreign_key(table):
    fk_d = dict()
    for fk in table['foreign_keys']:
        if fk[0] not in fk_d.keys():
            fk_d[fk[0]] = []
        fk_d[fk[0]].append(fk[1])

    for key in fk_d.keys():
        for col in fk_d[key]:
            if col in fk_d.keys():
                for c in fk_d[col]:
                    if [key,c] not in table['foreign_keys']:
                        table['foreign_keys'].append([key,c])
    return table


def enlarge_network(net_work,table):
    def in_side_enlarge(t1t2_fk,table_1,table_2,table):
        t1t2_fk = copy.deepcopy(t1t2_fk)
        t1t2_fk.sort()
        new_fk = []
        for fk in table['foreign_keys']:
            fk = copy.deepcopy(fk)
            fk.sort()
            if fk != t1t2_fk and ((table['column_names'][fk[0]][0] == table_1 and table['column_names'][fk[1]][0] == table_2) or (table['column_names'][fk[0]][0] == table_2 and table['column_names'][fk[1]][0] == table_1)):
                new_fk.append(fk)
        return new_fk

    def check_same_net(new_net,all_net_work):
        add_new_net = True
        for net_all in all_net_work:
            if len(net_all[1]) == len(new_net[1]):
                tf = False
                for fk in net_all[0]:
                    if fk not in new_net[0]:
                        tf = True
                        break
                if not tf:
                    add_new_net = False
                    break
        return add_new_net

    def add_table_to_enlarge(net,idx,table_1,table_2,table,new_net_work):
        for i,t in enumerate(table['table_names_original']):
            if i not in net[1]:
                for fk in table['foreign_keys']:
                    if (table['column_names'][fk[0]][0] == table_1 and table['column_names'][fk[1]][0] == i) or (table['column_names'][fk[0]][0] == i and table['column_names'][fk[1]][0] == table_1):
                        for fk2 in table['foreign_keys']:
                            if (table['column_names'][fk2[0]][0] == table_2 and table['column_names'][fk2[1]][0] == i) or (table['column_names'][fk2[0]][0] == i and table['column_names'][fk2[1]][0] == table_2):
                                new_net = copy.deepcopy(net)
                                new_net[1].insert(idx,i)
                                new_net[0][idx-1]=fk
                                new_net[0].insert(idx,fk2)
                                add_new_net = True
                                if check_same_net(new_net,new_net_work):
                                    new_net_work.append(new_net)
        return new_net_work
    
    def return_net_work_2(net_work):
        new_net = []
        for net in net_work:
            if len(net[1]) == 2:
                new_net.append(net)
        return new_net

    # inside enlarge (replace fk to enlarge) ::::::::::::::::
    new_net_work = copy.deepcopy(net_work)
    for net in net_work:
        if len(net[1]) == 1:
            continue
        else:
            # in-side enlarge:
            for i in range(1,len(net[1]),1):
                new_fk = in_side_enlarge(net[0][i-1],net[1][i-1],net[1][i],table)
                if new_fk:
                    for fk in new_fk:
                        new_net = copy.deepcopy(net)
                        new_net[0][i-1] = fk
                        new_net_work.append(new_net)
    
    # add table as bridge enlarge::::::::::::::::
    net_work = copy.deepcopy(new_net_work)
    for net in net_work:
        if len(net[1]) == 1:
            continue
        else:
            # add table to enlarge:
            for i in range(1,len(net[1]),1):
                new_net_work = add_table_to_enlarge(net,i,net[1][i-1],net[1][i],table,new_net_work)
    
    # delete too long table:
    for i in reversed(range(len(new_net_work))):
        if len(new_net_work[i][1]) > 5:
            del new_net_work[i]

    # two direction enlarge (maximun 3 tables), such as: [3,1] + [3,2] = [3,1,2]
    new_net_3 = []
    new_net = return_net_work_2(new_net_work)
    for i,net in enumerate(new_net): # 3 tables network
        for j,net2 in enumerate(new_net):
            if i < j and ((net[1][1] in net2[1] and net[1][0] not in net2[1]) or (net[1][0] in net2[1] and net[1][1] not in net2[1])):
                tmp_n3 = [[net[0][0],net2[0][0]],[net[1][0],net[1][1],net2[1][0] if net2[1][1] in net[1] else net2[1][1]]]
                if check_same_net(tmp_n3,new_net_work):
                    new_net_work.append(tmp_n3)
                    new_net_3.append(tmp_n3)

    # reversed direction enlarge (maximun 4 tables), such as: [3,1] + [2,1] = [3,1,2] 
    new_net = return_net_work_2(new_net_work)
    for i,net in enumerate(new_net): # 3 tables network
        for j,net2 in enumerate(new_net):
            if i < j and net[1][1] == net2[1][1] and net2[1][0] not in net[1]:
                tmp_n3 = [[net[0][0],net2[0][0]],[net[1][0],net[1][1],net2[1][0]]]
                if check_same_net(tmp_n3,new_net_work):
                    new_net_work.append(tmp_n3)
                    new_net_3.append(tmp_n3)
    for i,net in enumerate(new_net): # 4 tables network
        for j,net2 in enumerate(new_net_3):
            if (net2[1][-1] == net[1][0] and net[1][1] not in net2[1]) or (net2[1][-1] == net[1][1] and net[1][0] not in net2[1]):
                tmp_n4 = [[net2[0][0],net2[0][1],net[0][0]],[net2[1][0],net2[1][1],net2[1][2],net[1][1] if net2[1][-1] == net[1][0] else net[1][0]]]
                if check_same_net(tmp_n4,new_net_work):
                    new_net_work.append(tmp_n4)


    # add table to left to enlarge::::::::::::::::
    net_work = copy.deepcopy(new_net_work)
    for net in net_work:
        if len(net[1]) == 1 or len(net[1]) >= 5:
            continue
        else:
            # add table to enlarge:
            table_idx = net[1][0]
            for i,t in enumerate(table['table_names_original']):
                if i not in net[1]:
                    for fk in table['foreign_keys']:
                        if (table['column_names'][fk[0]][0] == table_idx and table['column_names'][fk[1]][0] not in net[1]):
                            new_tb_idx = table['column_names'][fk[1]][0]
                        elif (table['column_names'][fk[0]][0] not in net[1] and table['column_names'][fk[1]][0] == table_idx):
                            new_tb_idx = table['column_names'][fk[0]][0]
                        else:
                            continue
                        tmp_n4 = copy.deepcopy(net)
                        tmp_n4[0].insert(0,fk)
                        tmp_n4[1].insert(0,new_tb_idx)
                        if check_same_net(tmp_n4,new_net_work):
                            new_net_work.append(tmp_n4)

    return new_net_work



def build_index_range(table):
    table['index_range'] = []
    last_table_index = 0
    last_index_range = [0, len(table['column_names_original'])-1]
    for idx, col in enumerate(table['column_names_original']):
        if col[0] == last_table_index and last_index_range[0] == 0:
            last_index_range[0] = idx
        if col[0] > last_table_index:
            if last_index_range[0] == 1:
                last_index_range[0] = 0
            table['index_range'].append([last_index_range[0],idx - 1])
            last_index_range[0] = idx
            last_table_index += 1
    table['index_range'].append(last_index_range)
    return table






def build_super_column_name(table):
    table['super_column_names'] = copy.deepcopy(table['column_names'])
    table['super_table_column_names_original'] = copy.deepcopy(table['table_column_names_original'])
    assert len(table['primary_keys']) <= len(table['table_names'])
    for idx in range(len(table['column_names'])):

        col = table['super_column_names'][idx]
        t_col  = table['super_table_column_names_original'][idx]
        # * -> all
        col[1] = col[1].replace('*','all')
        
        # primary key
        if idx in table['primary_keys']:
            col[1] += " ( primary key )"
        
        col += [[0,idx]]
        t_col += [[0,idx]]

        if t_col[1].endswith('*'):
            col_ar = [col[0], AGG_LANGUAGE[3]+' '+col[1], [3,idx]]
            table['super_column_names'].append(col_ar)
            col_ar = [t_col[0], AGG_OPS[3]+'('+t_col[1]+')', [3,idx]]
            table['super_table_column_names_original'].append(col_ar)

            col_ar = [col[0], AGG_LANGUAGE[1]+' '+col[1], [1,idx]]
            table['super_column_names'].append(col_ar)
            col_ar = [t_col[0], AGG_OPS[1]+'('+t_col[1]+')', [1,idx]]
            table['super_table_column_names_original'].append(col_ar)

            col_ar = [col[0], AGG_LANGUAGE[2]+' '+col[1], [2,idx]]
            table['super_column_names'].append(col_ar)
            col_ar = [t_col[0], AGG_OPS[2]+'('+t_col[1]+')', [2,idx]]
            table['super_table_column_names_original'].append(col_ar)
        else:
            if table['new_column_types'][idx] == 'number':
                for i in range(1,len(AGG_OPS)):
                    col_ar = [col[0], AGG_LANGUAGE[i]+' '+col[1], [i,idx]]
                    table['super_column_names'].append(col_ar)
                    col_ar = [t_col[0], AGG_OPS[i]+'('+t_col[1]+')', [i,idx]]
                    table['super_table_column_names_original'].append(col_ar)
            else:
                col_ar = [col[0], AGG_LANGUAGE[3]+' '+col[1], [3,idx]]
                table['super_column_names'].append(col_ar)
                col_ar = [t_col[0], AGG_OPS[3]+'('+t_col[1]+')', [3,idx]]
                table['super_table_column_names_original'].append(col_ar)
    return table






def check_col_table_similarity(_tokenizer, table_names, col_token):
    col_lemma = " ".join([tok.lemma_ for tok in col_token])
    if col_lemma == table_names or " ".join([tok.text for tok in col_token]) == table_names:
        return True
    table_token = _tokenizer.tokenize(table_names)
    if table_token[-1].lemma_ == col_token[0].lemma_ or col_lemma == " ".join([tok.lemma_ for tok in table_token]) or table_token[-1].lemma_ == col_token[0].text:
        return True
    return False


def create_mini_network(network,table):
    table_all = []
    tmp_net = []
    for n in network:
        if len(n[1]) > 1:
            to_all_table = copy.deepcopy(n[1])
            to_all_table.sort()
            if to_all_table not in table_all:
                table_all.append(to_all_table)

    for new in network:
        if len(new[0]) == 2 and len(new[1]) == 3:
            if new[0][0][1] in new[0][1] or new[0][0][0] in new[0][1]:
                new_net = None
                if new[0][0][1] in new[0][1] and table['column_names'][new[0][0][1]][0] == new[1][1]:
                    if new[0][0][1] == new[0][1][0]:
                        new_net = [[[new[0][0][0],new[0][1][1]]],[new[1][0],new[1][2]]]
                    else:
                        new_net = [[[new[0][0][0],new[0][1][0]]],[new[1][0],new[1][2]]]
                elif new[0][0][0] in new[0][1] and table['column_names'][new[0][0][0]][0] == new[1][1]:
                    if new[0][0][0] == new[0][1][0]:
                        new_net = [[[new[0][0][1],new[0][1][1]]],[new[1][0],new[1][2]]]
                    else:
                        new_net = [[[new[0][0][1],new[0][1][0]]],[new[1][0],new[1][2]]]
                if new_net:
                    to_all_table = copy.deepcopy(new_net[1])
                    to_all_table.sort()
                    if to_all_table not in table_all:
                        table_all.append(to_all_table)
                        tmp_net.append(new_net)
    network.extend(tmp_net)
    return network


def re_identify_boolean_type(tables,use_extra_col_types,database_path):
    for db_i,table in enumerate(tables):
        print()
        print(tables[db_i]['db_id'] + " " + str(db_i))
        try:
            db_ = DBEngine(table,database_path)
        except:
            continue
        
        tables[db_i]['data_samples'] = [[]]
        for t in range(len(table['table_names'])):
            data_samples = db_.col_data_samples(t)
            tables[db_i]['data_samples'].extend(data_samples)

        all_cols = []
        for t in range(len(table['table_names'])):
            cols,all_col = db_.db_col_type_check(t)
            all_cols.extend(all_col)
        assert len(all_cols) == len(tables[db_i]['column_types']) - 1
        tables[db_i]['column_types_checked'] = ["None"] * len(tables[db_i]['column_types'])
        
        for i,col in enumerate(all_cols):
            if col[1] == 0:  # boolean
                if tables[db_i]['column_types'][i+1] != "boolean":
                    print(tables[db_i]['column_names'][i+1][1]+" "+tables[db_i]['column_types'][i+1]+" --> boolean")
                tables[db_i]['column_types'][i+1] = "boolean"
                tables[db_i]['column_types_checked'][i+1] = "boolean"
            elif col[1] == 1: # text
                if tables[db_i]['column_types'][i+1] != "text":
                    print(tables[db_i]['column_names'][i+1][1]+" "+tables[db_i]['column_types'][i+1]+" --> text")
                if " or " in tables[db_i]['column_names'][i+1][1]:
                    tables[db_i]['column_types'][i+1] = "boolean"
                    tables[db_i]['column_types_checked'][i+1] = "boolean"
                else:
                    tables[db_i]['column_types'][i+1] = "text"
                    tables[db_i]['column_types_checked'][i+1] = "text"
            elif col[1] == 2: # number
                if tables[db_i]['column_types'][i+1] != "number":
                    print(tables[db_i]['column_names'][i+1][1]+" "+tables[db_i]['column_types'][i+1]+" --> number")
                tables[db_i]['column_types'][i+1] = "number"
                tables[db_i]['column_types_checked'][i+1] = "number"
            elif col[1] == 3: # DATE TIME
                if tables[db_i]['column_types'][i+1] != "time":
                    print(tables[db_i]['column_names'][i+1][1]+" "+tables[db_i]['column_types'][i+1]+" --> time")
                tables[db_i]['column_types'][i+1] = "time"
                tables[db_i]['column_types_checked'][i+1] = "time"
            elif col[1] == 4: # YEAR
                if tables[db_i]['column_types'][i+1] != "year":
                    print(tables[db_i]['column_names'][i+1][1]+" "+tables[db_i]['column_types'][i+1]+" --> year")
                if use_extra_col_types:
                    tables[db_i]['column_types'][i+1] = "year"
                    tables[db_i]['column_types_checked'][i+1] = "year"
                else:
                    tables[db_i]['column_types'][i+1] = "number"
                    tables[db_i]['column_types_checked'][i+1] = "number"
            else:
                pass

    return tables


def remove_start_table(tables,schemas):
    for i,table,schema in zip(range(len(tables)),tables,schemas):
        tables[i]["old_column_names"] = copy.deepcopy(tables[i]["column_names"])
        for j,col in enumerate(table["column_names"]):
            if col[0] >= 0  and (col[1].startswith(schema.table_tokens_lemma_str[col[0]]+" ") or col[1].startswith(schema.table_tokens_text_str[col[0]]+" ")):
                tables[i]["column_names"][j][1] = " ".join(tables[i]["column_names"][j][1].split(" ")[schema.table_tokens_lemma_str[col[0]].count(" ")+1:])
            elif j in table["primary_keys"] and tables[i]["column_names"][j][1].endswith(" id") and schema.table_tokens_lemma_str[col[0]].startswith(tables[i]["column_names"][j][1][0:-3]):
                tables[i]["column_names"][j][1] = tables[i]["column_names"][j][1][-2:]
            elif (tables[i]["column_names"][j][1] == schema.table_tokens_lemma_str[col[0]] or tables[i]["column_names"][j][1] == schema.table_tokens_text_str[col[0]]) and tables[i]["column_types"][j] == "text" and "name" not in schema.table_col_lemma[col[0]]:
                tables[i]["column_names"][j][1] = tables[i]["column_names"][j][1] + " | name"
    return tables




def recover_table_name(tables):
    for i,table in enumerate(tables):
        try_to_add = []
        for j,col,ocol in zip(range(len(table["column_names"])),table["column_names"],table["old_column_names"]):
            if col[0] >= 0  and  " | " not in col[1] and ocol[1].count(" ") > col[1].count(" "):
                try_to_add.append(col[0])
        for j,col,ocol in zip(range(len(table["column_names"])),table["column_names"],table["old_column_names"]):
            if col[0] >= 0  and  " | " not in col[1] and ocol[1].count(" ") > col[1].count(" ") and try_to_add.count(col[0]) == 1:
                print(tables[i]["column_names"][j][1])
                tables[i]["column_names"][j][1] = ocol[1]
                print(tables[i]["column_names"][j][1])
                print()
        tables[i].pop("old_column_names")
    return tables


def unifie_words(tables):
    WORDS = [
        ["enrolment","enrollment"],
        ["enroll","enrol"],
        ["lives in","lived in"],
        ["live in","lived in"]
    ]
    for i,table in enumerate(tables):
        for j, col in enumerate(table["column_names"]):
            for w in WORDS:
                if w[0] == col[1] or col[1].startswith(w[0] + " ") or col[1].endswith( " " + w[0]) or " " + w[0] + " " in col[1]:
                    table["column_names"][j][1] = col[1].replace(w[0],w[1])
    
        for j, tb in enumerate(table["table_names"]):
            for w in WORDS:
                if w[0] == tb or tb.startswith(w[0] + " ") or tb.endswith( " " + w[0]) or " " + w[0] + " " in tb:
                    table["table_names"][j] = tb.replace(w[0],w[1])
    return tables


def analyse_same_column(tables,schemas,database_path):
    def they_are_same(col_1_idx,col_2_idx,table,schema,all_pair,db_):
        def contain_table_name(col_1_idx,col_2_idx,table,schema):
            for ttls in schema.table_tokens_lemma_str[schema.column_names_original[col_1_idx][0]].split(" "):
                for ttls2 in schema.table_tokens_lemma_str[schema.column_names_original[col_2_idx][0]].split(" "):
                    if ttls == ttls2:
                        return True

            for ttls in schema.table_tokens_lemma_str:
                for t in ttls.split(" "):
                    if t in schema.column_tokens_lemma_str[col_1_idx] or t in schema.column_tokens_lemma_str[col_2_idx]:
                        return True
            return False

        if [col_1_idx,col_2_idx] in all_pair or [col_2_idx,col_1_idx] in all_pair:
            return True
        if " " not in schema.column_tokens_lemma_str[col_1_idx] and " " not in schema.column_tokens_lemma_str[col_2_idx]:
            return False
        if col_1_idx != col_2_idx:
            if schema.column_tokens_lemma_str[col_1_idx] == schema.column_tokens_lemma_str[col_2_idx]:
                pass
            elif schema.table_tokens_lemma_str[table['column_names'][col_1_idx][0]] + schema.column_tokens_lemma_str[col_1_idx] == schema.column_tokens_lemma_str[col_2_idx]:
                pass
            elif  schema.column_tokens_lemma_str[col_1_idx] == schema.table_tokens_lemma_str[table['column_names'][col_2_idx][0]] + schema.column_tokens_lemma_str[col_2_idx]:
                pass
            elif schema.table_tokens_lemma_str[table['column_names'][col_1_idx][0]] + schema.column_tokens_text_str[col_1_idx] == schema.column_tokens_text_str[col_2_idx]:
                pass
            elif  schema.column_tokens_text_str[col_1_idx] == schema.table_tokens_text_str[table['column_names'][col_2_idx][0]] + schema.column_tokens_lemma_str[col_2_idx]:
                pass
            elif schema.table_tokens_text_str[table['column_names'][col_1_idx][0]] + schema.column_tokens_lemma_str[col_1_idx] == schema.column_tokens_lemma_str[col_2_idx]:
                pass
            elif  schema.column_tokens_lemma_str[col_1_idx] == schema.table_tokens_text_str[table['column_names'][col_2_idx][0]] + schema.column_tokens_lemma_str[col_2_idx]:
                pass
            elif  schema.column_tokens_text_str[col_1_idx] == schema.column_tokens_text_str[table['column_names'][col_2_idx][0]] + schema.column_tokens_text_str[col_2_idx]:
                pass
            elif schema.column_tokens_text_str[table['column_names'][col_1_idx][0]] + schema.column_tokens_text_str[col_1_idx] == schema.column_tokens_text_str[col_2_idx]:
                pass
            else:
                return False
            if schema.column_types[col_1_idx] == schema.column_types[col_2_idx] and contain_table_name(col_1_idx,col_2_idx,table,schema) and ((db_ and db_.db_content_are_same(col_1_idx,col_2_idx)) or db_ is None):
                print(schema.column_tokens_lemma_str[col_1_idx] + " " + schema.column_tokens_lemma_str[col_2_idx])
                return True
        return False
        
    for it, table,schema in zip(range(len(tables)),tables,schemas):
        same_col_idxs = []
        all_pair = []
        try:
            db_ = DBEngine(table,database_path)
        except:
            db_ = None
        for i,col in enumerate(table["column_names"]):
            same_col_idx = []
            for j,col in enumerate(table["column_names"]):
                if they_are_same(i,j,table,schema,all_pair,db_):
                    same_col_idx.append(j)
                    all_pair.append([i,j])
            same_col_idxs.append(same_col_idx)
        tables[it]["same_col_idxs"] = same_col_idxs 
    return tables


def seperate_col_name(tables,all_words,schemas):
    suffix = {
        "ster","sters","ages","bility","cracy","doms","facture","faction","hood","ices","ions","ours","ship","tude","graphy","graphs","nomy","hood","ants","ment","some","ward","like","fold","most","ways","meter","miter","gram",# "proof","able"
    }
    for i,table,schema in zip(range(len(tables)),tables,schemas):
        for j,col in enumerate(table["column_names"]):
            bk_bool = False
            if col[0] >= 0  and " " not in col[1] and len(col[1]) > 8:
                if col[1] not in all_words:
                    for x in range(4,len(col[1])-3):

                        if col[1][0:x] in all_words and col[1][x:] in all_words and col[1][0:x] not in STOP_WORDS and col[1][x:] not in STOP_WORDS and col[1][x:] not in suffix:
                            print(tables[i]["column_names"][j][1] + " ==================> " + tables[i]["column_names"][j][1] + " | " + col[1][0:x] + " " + col[1][x:])
                            tables[i]["column_names"][j][1] += " | " + col[1][0:x] + " " + col[1][x:]
                            bk_bool = True
                            break
                if not bk_bool and schema.column_tokens_lemma_str[j] != col[1] and len(schema.column_tokens_lemma_str[j]) > 8 and schema.column_tokens_lemma_str[j] not in all_words: 
                    col_lemma = schema.column_tokens_lemma_str[j]
                    for x in range(4,len(schema.column_tokens_lemma_str[j])-3):
                        if col_lemma[0:x] in all_words and col_lemma[x:] in all_words and col_lemma[0:x] not in STOP_WORDS and col_lemma[x:] not in STOP_WORDS and col_lemma[x:] not in suffix:
                            print(tables[i]["column_names"][j][1] + " ==================> " + tables[i]["column_names"][j][1] + " | " + col_lemma[0:x] + " " + col_lemma[x:])
                            tables[i]["column_names"][j][1] += " | " + col_lemma[0:x] + " " + col_lemma[x:]
                            break
    return tables

def add_line_break(sql):
    # add "\n" after each ","
    sql = sql.replace(",", ",\n")
    sql = sql.replace('",\n', '",') # fix multi-column pk
    # add "\n" after the first "("
    sql = sql[:sql.find("(")+1] + "\n" + sql[sql.find("(")+1:]
    # add "\n" before the last ")"
    sql = sql[:sql.rfind(")")] + "\n" + sql[sql.rfind(")"):]
    return sql

def correct_primary_keys(tables,schemas,database_path):
    for it, table,schema in zip(range(len(tables)),tables,schemas):
        table['original_primary_keys'] = copy.deepcopy(table['primary_keys'])
        same_col_idxs = []
        all_pair = []
        try:
            db_ = DBEngine(table,database_path)
            db_infos = db_.get_db_structure_info()
        except:
            db_ = None
            continue
        if db_infos:
            for db_info in db_infos:
                create_table_sql = db_info[4]
                if "\n" not in create_table_sql:
                    create_table_sql = add_line_break(create_table_sql)
                lines = create_table_sql.split('\n')
                find_pk = False
                for line in lines:
                    line = line.strip().replace("\t"," ").replace("  "," ").replace("  "," ").replace("  "," ").replace("  "," ")
                    if ' --' in line:
                        line = line.split('--')[0]
                    if line.strip().upper().endswith(" NOT NULL,"):
                        line = line[:-9]
                    if line.strip().upper().endswith(" NOT NULL ,"):
                        line = line[:-10]
                    if line.strip().upper().endswith(" NOT NULL"):
                        line = line[:-8]
                    if line.strip().lower().endswith(" autoincrement"):
                        line = line[:-14]
                    if line.strip().lower().endswith(" autoincrement,"):
                        line = line[:-15]
                    if line.strip().lower().endswith(" autoincrement ,"):
                        line = line[:-16]
                    if line.strip().upper().endswith(" NOT NULL,"):
                        line = line[:-9]
                    if line.strip().upper().endswith(" NOT NULL ,"):
                        line = line[:-10]
                    if line.strip().upper().endswith(" NOT NULL"):
                        line = line[:-8]
                    if line.strip().upper().endswith(" PRIMARY KEY,") or line.strip().upper().endswith(" PRIMARY KEY ,") or line.strip().upper().endswith(" PRIMARY KEY"):
                        while True:                                
                            if line[0].isalpha():
                                break
                            else:
                                line = line[1:]
                        ls = line.split(" ")
                        col = (db_info[1]+'.'+ "".join([t if t.isalpha() or t == '_' else "" for t in ls[0]])).lower()
                        assert col in schema.original_table["tc_fast"]
                        p_id = schema.original_table["tc_fast"].index(col)
                        assert p_id in schema.primaryKey
                        find_pk = True
                    elif "PRIMARY KEY" in line.upper():
                        line = line.lower().strip().replace(" key("," key (").replace(" key,"," key ,")
                        ls = line.split(" ")
                        assert 'primary' in ls and 'key' in ls
                        for i,l in enumerate(ls):
                            if i > 0 and ls[i-1] == 'primary' and l == 'key':
                                key_str = " ".join(ls[i+1:])
                        find_pk = True
                        if ',' in key_str:
                            rp_keys = []
                            for k in key_str.split(','):
                                col = db_info[1].lower()+'.'+"".join([t if t.isalpha() or t == '_' else "" for t in k])
                                if col and col in schema.original_table["tc_fast"]:
                                    rp_keys.append(col)
                            if len(rp_keys) > 1:
                                for col in rp_keys:
                                    if col in schema.original_table["tc_fast"]:
                                        p_id = schema.original_table["tc_fast"].index(col)
                                        if p_id in schema.primaryKey:
                                            del(table['primary_keys'][table['primary_keys'].index(p_id)])
                if not find_pk:
                    if db_info[1].lower() in schema.table_names_original:
                        tid = schema.table_names_original.index(db_info[1].lower())
                        for col in schema.primaryKey:
                            if schema.column_tokens_table_idx[col] == tid:
                                print("EEEEE")


def label_disjoint_tables(tables,database_path):
    for it, table in enumerate(tables):
        table['unique_fk'] = []
        db_ = DBEngine(table,database_path)
        for fk in table['foreign_keys']:
            if fk[0] not in table['primary_keys'] and fk[0] not in table['unique_fk']:
                try:
                    if not db_.check_disjoint_column(fk[0]):
                        table['unique_fk'].append(fk[0])
                except:
                    continue
            if fk[1] not in table['primary_keys'] and fk[1] not in table['unique_fk']:
                try:
                    if not db_.check_disjoint_column(fk[1]):
                        table['unique_fk'].append(fk[1])
                except:
                    continue


def bridge_table_for_many2many_relationship(tables):
    for it, table in enumerate(tables):
        table['bridge_table'] = []
        table['many2many'] = dict()
        for net in table['network']:
            net2dict = dict()
            if len(net[1]) == 3 and len(net[0]) == 2 and net[0][0][0] not in net[0][1] and net[0][0][1] not in net[0][1]:
                for fks in net[0]:
                    if fks[0] not in table['primary_keys']:
                        t0 = table['column_names'][fks[0]][0]
                        net2dict[t0] = 1 if t0 not in net2dict else net2dict[t0] + 1
                    if fks[1] not in table['primary_keys']:
                        t1 = table['column_names'][fks[1]][0]
                        net2dict[t1] = 1 if t1 not in net2dict else net2dict[t1] + 1
                count = 0
                for key in net2dict:
                    if net2dict[key] > 1:
                        table['bridge_table'].append(key)
                        count += 1
                
                if count == 1 and table['bridge_table'][-1] not in table['many2many']:
                    table['many2many'][table['bridge_table'][-1]] = []
                    for t in net[1]:
                        if t != table['bridge_table'][-1]:
                            table['many2many'][table['bridge_table'][-1]].append(t)
                    


if __name__ == '__main__':
    # 1. Hyper parameters
    args = construct_hyper_param()
    database_path = args.db_path

    # 2. Prepare data
    tables = json.load(open(args.in_file,'r'))
    all_words = pickle.load(open(os.path.join("./NatSQL/data/20k.pkl"), 'rb'))
    new_tables = []

    lstem = MyStemmer()
    _tokenizer = get_spacy_tokenizer()
    schemas = []
    for table in tables:
        schemas.append(Schema_Token(_tokenizer,lstem,table,None))

    if args.add_debug_col:
        for t in tables:
            i = 0
            t['column_names_order'] = copy.deepcopy(t['column_names']) 
            for c in t['column_names_order']:
                c[0] = c[0]*1000 + i
                i += 1
    else:
        if 'column_names_order' in tables[0].keys():
            for t in tables:
                t.pop('column_names_order')
    
    if args.correct_primary_keys:
        correct_primary_keys(tables,schemas,database_path)
        label_disjoint_tables(tables,database_path)
    
    if args.correct_col_type:
        tables = re_identify_boolean_type(tables,args.use_extra_col_types,database_path)
        tables = unifie_words(tables)

    if args.remove_start_table:
        tables = remove_start_table(tables,schemas)

    if args.analyse_same_column:
        tables = analyse_same_column(tables,schemas,database_path)
    
    if args.seperate_col_name:
        tables = seperate_col_name(tables,all_words,schemas)

    if args.table_transform:
        for table,schema in zip(tables,schemas):
            nt = table_transform(table,args,schema)
            nt = build_index_range(nt)
            nt = expand_foreign_key(nt)
            nt = build_table_network(nt)
            # enlarge network is not work well with NatSQL version >= 1.3
            nt["network"] = enlarge_network(nt["network"],nt)

            nt["tc_fast"] = []
            for tctc in nt["table_column_names_original"]:
                nt["tc_fast"].append(tctc[1].lower())
            nt["table_orig_low"] = []
            for table_orig_low in nt["table_names_original"]:
                nt["table_orig_low"].append(table_orig_low.lower())
            if 'unique_fk' in nt:
                for i in range(len(nt['unique_fk'])):
                    nt['unique_fk'][i] = reversed_link_back_col(nt['unique_fk'][i], nt)
            new_tables.append(nt)

        if args.correct_primary_keys:
            bridge_table_for_many2many_relationship(new_tables)

        if "old_column_names" in new_tables[0] and args.recover_previous_column_content:
            pass
        elif "old_column_names" in new_tables[0]:
            for i,table in enumerate(new_tables):
                new_tables[i].pop("old_column_names")

        if args.add_star_on_first_col:
            for i,table in enumerate(new_tables):
                new_tables[i]["column_names"].insert(0,[-1,"*"])
                new_tables[i]["column_names_original"].insert(0,[-1,"*"])


        json.dump(new_tables,open(args.out_file,'w'), indent=2)
    else:
        for table in tables:
            if args.keepOriginal:
                if "same_col_idxs" in table:
                    table.pop("same_col_idxs")
                if "old_column_names" in table:
                    table.pop("old_column_names")
            else:
                table['table_column_names_original'] = []
                table['link_back'] = []
                for it,item in enumerate(table['column_names_original']):
                    table['link_back'].append([it,it])
                    if item[0] >= 0:
                        table['table_column_names_original'].append([item[0], table['table_names_original'][item[0]]+"."+item[1] ] )
                    else:
                        table['table_column_names_original'].append(item )
        json.dump(tables,open(args.out_file,'w'), indent=2)
    
    