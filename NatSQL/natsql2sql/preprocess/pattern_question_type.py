from .others_pattern import create_pattern_toks

def pattern_for_skip(pattern,sentence_ts,table_match,col_match,entt,schema,db_match,table_idx,negative):
    return 5,0,0,pattern

def pattern_to_select(pattern,sentence_ts,table_match,col_match,entt,schema,db_match,table_idx,negative):
    # force to change the type to 0
    return 0,0,0,pattern

def pattern_to_continue(pattern,sentence_ts,table_match,col_match,entt,schema,db_match,table_idx,negative):
    # set(change) the following sentence to the patter (type) as now.
    return 1,0,0,pattern

def pattern_to_follow_last(pattern,sentence_ts,table_match,col_match,entt,schema,db_match,table_idx,negative):
    # set the pattern (type) now following the previous type
    return 2,0,0,pattern

def pattern_to_new_where(pattern,sentence_ts,table_match,col_match,entt,schema,db_match,table_idx,negative):
    # force to change the type to 1 or other where condition type
    return 3,0,0,pattern

def pattern_for_IN_TABLE(pattern,sentence_ts,table_match,col_match,entt,schema,db_match,table_idx,negative):
    return 4,0,0,pattern

def pattern_for_substring(pattern,sentence_ts,table_match,col_match,entt,schema,db_match,table_idx,negative):
    return 6,0,0,pattern

def pattern_for_age(pattern,sentence_ts,table_match,col_match,entt,schema,db_match,table_idx,negative):
    return 7,0,0,pattern

def pattern_for_which(pattern,sentence_ts,table_match,col_match,entt,schema,db_match,table_idx,negative):
    return 8,0,0,pattern

def pattern_for_must_change(pattern,sentence_ts,table_match,col_match,entt,schema,db_match,table_idx,negative):
    return 9,0,0,pattern

def pattern_to_follow_last2(pattern,sentence_ts,table_match,col_match,entt,schema,db_match,table_idx,negative):
    return 10,0,0,pattern

PATTERNS = [
    'START SEARCH DATABASE',
    
    # 8 # which + table(col) . + second sentence
    "which TABLE * COL | which TABLE COL | which TABLE | which COL | which COL of TABLE | which | who | what | which * | which TABLE V TABLE | which TABLE V"+\
    " | order TABLE * COL | order TABLE COL | order TABLE | order COL | order COL of TABLE | order COL of TABLE" +\
    " | find TABLE * COL | find TABLE COL | find TABLE | find COL | find COL of TABLE | find COL of TABLE | find *" +\
    " | display TABLE * COL | display TABLE COL | display TABLE | display COL | display COL of TABLE | display COL of TABLE | display *" +\
    " | list TABLE * COL | list TABLE COL | list TABLE | list COL | list COL of TABLE | list COL of TABLE | list *" +\
    " | what be TABLE * COL | what be TABLE COL | what be TABLE | what be COL | what be COL of TABLE | what be COL of TABLE | what be *",

    # 5: do nothing
    "+ NO# * IN each TABLE | + IN each TABLE | + NO# * IN each COL | + IN each COL | + be _ IN each COL | + be _ IN each TABLE"+\
        " | + IN each COL | + NO# * IN each COL | + * * IN each COL"+\
        " | + each COL | + NO# * each COL | + * * each COL | + TABLE IN each COL | + NO# * each * COL"+\
        " | + grouped by COL | + grouped by TABLE | + grouped by * COL | + grouped by * TABLE"+\
        " | + IN each COL * | + IN each TABLE * | + IN every TABLE | + IN every COL | + IN every TABLE * | + IN every COL *"+\
        " | + _ each COL | + IN each NN | + IN each TABLE TABLE | + IN each COL COL" + 
    ' | + have each TABLE | + have each COL | + V each TABLE | + V each COL | after DATE | before DATE', 

    # 0
    # select:
    # 'show',  'give', 'who', 'what + (the) + col', 'describe', 'what are', 'how', 'count', 'sort', 'find', 'where', 'whose', '【compute】',  'what is', 'list', "what 's",  'when',  '【which】',  'tell',  'return'
    # force to change the type to 0
    "$and what COL be | $and what COL * | $and what COL * * | $and what be _ | $and what be _ _ | $and what be _ _ _ | $and how many _ _ _ | $and how many _ _ | $and list _ _ _ | $and list _ _ | $and what 's _ _ _ | $and what 's _ _ | $and COL of each _ | $and what be _ _ _ _ | $and how many _ _ _ _ | $and list _ _ _ _ _ | $and what be _ _ _ _ _ | $and how many _ _ _ _ _ | $and list _ _ _ _ _ _ | $and what _ be _ _ | $and what _ be _ _ _ | $and what _ be _ _ _ _ | $and list _ _ _ _ _ _ _ | $and what be _ _ _ _ _ _ | $and what be _ _ _ _ _ _ _ | $and what be _ _ _ _ _ _ _ _ | $and show _ | $and show _ _ | $and show _ _ _ | $and show _ _ _ _ | $and find _ | $and find _ _ | $and find _ _ _ | $and find _ _ _ _ | $and when do _ _ | $and when do _ _ _ | $and when do _ _ _ _ | $and when be _ _ | $and when be _ _ _ | $and when be _ _ _ _ | $and how long _ | $and how long _ _ | $and how long _ _ _ | $and how long _ _ _ _ | $and how long _ _ _ _ _ | $and where be TABLE * ? | $and where be COL * ?| $and what COL * * * * ? | $and what COL * * * ?",#

    # 2
    # set the pattern (type) now following the previous type
    "from GRSM to GRSM | from SGRSM to SGRSM | + to COL | + in PCOL | + in * PCOL | from GRSM | to GRSM | from SGRSM | to SGRSM | _ and | + COL of TABLE | + number of COL | + number of COL for all COL | total COL | + COL of COL | + TABLE COL of COL | + COL IN TABLE | + COL IN COL | or NUM | and | COL and COL | COL and COL * | COL and COL $IN $DT * | TABLE COL and TABLE COL | TABLE COL and TABLE COL * | PDB_C | UPPER * * | UPPER * | + IN COL | + IN COL COL | include PDB | V for each COL and COL | IN NN | IN NNS"
    +" | from JJS to JJS | from SJJS to SJJS | NO# DB V | NO# DB COL" # NO# represent there is not #
    +" | IN * order | IN ascend order | IN descend order"# | IN ascend alphabetical order | IN descend alphabetical order | IN alphabetical order"
    +" | * average and | * average or | in total | on total | to DB",# | who TABLE | who * TABLE | who * * TABLE
    
    # 10
    # set the pattern (type) now following the previous type if previous type is not 0
    "GRSM than _ _ _ | GRSM than _ _ | GRSM than _ | GRSM NUM | GRSM AGG COL | GRSM TABLE | named DB | named PDB | named SDB",

    # 1
    # pattern_to_continue: 
    # set(change) the following sentence to the patten (type) as now.
    "+ with PCOL | + that have PCOL | + who have PCOL | + have PCOL | + with * PCOL | + that have * PCOL | + who have * PCOL | + have * PCOL"
    +" | + V * PCOL"+" | + V PCOL | + that PCOL | + who PCOL | + and PCOL | + and * PCOL | + PCOL | + * PCOL | and TABLE with PCOL"
    +" | + with TABLE PCOL | + that have TABLE PCOL | + who have TABLE PCOL | + have TABLE PCOL | + with * TABLE PCOL | + that have * TABLE PCOL | + who have * TABLE PCOL | + have * TABLE PCOL"
    +" | + V * TABLE PCOL"+" | + V TABLE PCOL | + that TABLE PCOL | + who TABLE PCOL | + and TABLE PCOL | + and * TABLE PCOL | + TABLE PCOL | + * TABLE PCOL"
    +" | + with _TABLE_ TCOL | + that have _TABLE_ TCOL | + who have _TABLE_ TCOL | + have _TABLE_ TCOL | + with * _TABLE_ TCOL | + that have * _TABLE_ TCOL | + who have * _TABLE_ TCOL | + have * _TABLE_ TCOL"
    +" | + V * _TABLE_ TCOL"+" | + V _TABLE_ TCOL | + that _TABLE_ TCOL | + who _TABLE_ TCOL | + and _TABLE_ TCOL | + and * _TABLE_ TCOL | + _TABLE_ TCOL | + * _TABLE_ TCOL"
    +" | + from which PCOL | + from which * PCOL | + from which TABLE PCOL | + from which * TABLE PCOL | + from which _TABLE_ TCOL | + from which * _TABLE_ TCOL"
    +" | + that be PCOL | + that be * PCOL | + that be TABLE PCOL | + that be * TABLE PCOL | + that be _TABLE_ TCOL | + that be * _TABLE_ TCOL"
    +" | + in PCOL | + in * PCOL | + in TABLE PCOL | + in PCOL of TABLE | + in * TABLE PCOL | + in * PCOL of TABLE | + in _TABLE_ TCOL | + in * _TABLE_ TCOL | + in TCOL of _TABLE_ | + in * TCOL of _TABLE_"
    +" | except those | except that | in the | in | order by"
    +" | PCOL be not | * PCOL be not | TABLE PCOL be not | * TABLE PCOL be not | _TABLE_ TCOL be not | * _TABLE_ TCOL be not"
    +" | WP NOT V | NOT V | WP NOT PCOL | NOT PCOL | WP NOT V * | NOT V * | WP have NOT V * | WP have NOT V"
    +" | + that IN * TABLE | + that IN TABLE | + that be * TABLE | + that be TABLE | + that V IN TABLE | + that * TABLE | + that COL TABLE"
    +" | + that IN * PTABLE PTABLE | + that IN PTABLE PTABLE"
    +" | + do TABLE with COL | what is the"
    +" | + PCOL * | be NOT V",# +" | IN DB",


    # 3
    # force to change the type 0 to 1 or other where condition type
    "whose _ _ _ _ | whose _ _ _ | whose _ _ | whose _ _ _ _ _ | but _ _ _ _ | but _ _ _ | but _ _ | but _ | or _ _ _ _ | or _ _ _ | or _ _ | or _ | DB | which _ be _ | and have TABLE | and COL DB | at least _ _ _ | at least _ _ _ _ | at least _ _ | at most _ _ _ | at most _ _ _ _ | at most _ _"\
    +" | TABLE DB | COL DB | @ 1&SGRSM 1&than | @ 1&GRSM 1&than | @ 1&SGRSM 1&NUM | @ 1&GRSM 1&NUM",

    # 4
    "+ -that IN * TABLE | + -that IN TABLE | + -that be * TABLE | + -that be TABLE | + -that V IN TABLE | + -that * TABLE | + -that COL TABLE"
    +" | + -that IN * PTABLE PTABLE | + -that IN PTABLE PTABLE",
    
    # 6
    "substring | IN substring | IN AGG | ascend | descend | * ascend _ | * descend _",

    # 9 100% set(change) the following sentence to the patten (type) as now.
    "+ with AGG PCOL | + with AGG * PCOL | + with AGG TABLE PCOL | + with * AGG TABLE PCOL | + with AGG _TABLE_ TCOL | + with * AGG _TABLE_ TCOL"
    +" | + with AGG AGG PCOL | + with AGG AGG * PCOL | + with AGG AGG TABLE PCOL | + with * AGG AGG TABLE PCOL | + with AGG AGG _TABLE_ TCOL | + with * AGG AGG _TABLE_ TCOL",
    
]

PATTERN_FUN = [
    None,
    pattern_for_which,      # return 8
    pattern_for_skip, # skip these pattern
    pattern_to_select, # make type to 0
    pattern_to_follow_last, # make type now followed last type
    pattern_to_follow_last2,
    pattern_to_continue, # make next type follow type now
    pattern_to_new_where,
    pattern_for_IN_TABLE,
    pattern_for_substring,
    pattern_for_must_change,
]

PATTERNS_TOKS = create_pattern_toks([], PATTERNS)





############# FOR ADD COL:
ADD_COL_PATTERNS = [
    "+ NO# * IN each TABLE | + IN each TABLE | NO# * IN each COL | + IN each COL | + be _ IN each COL | + be _ IN each TABLE"+\
        " | + IN each COL | + NO# * IN each COL | + NO# * * IN each COL"+\
        " | + each COL | + NO# * each COL | + NO# * * each COL | + TABLE IN each COL | + NO# * each * COL"+\
        " | + grouped by COL | + grouped by TABLE | + grouped by * COL | + grouped by * TABLE"+\
        " | + IN each COL * | + IN each TABLE * | + IN every TABLE | + IN every COL | + IN every TABLE * | + IN every COL *"+\
        " | + _ each COL | + IN each NN | + have each TABLE | + have each COL | + V each TABLE | + V each COL | + V by each TABLE | + V by each COL | + IN each TABLE TABLE | + IN each COL COL | + IN each COL of TABLE",
    
    "under NUM | above NUM | over NUM | * under NUM | * above NUM | * over NUM",

    "@ 2&SGRSM | @ 2&SJJS",
    "@ 1&SGRSM | @ 1&SJJS", # return 1
    "@ 1&DATE | @ 2&YEAR",
    "@ 1&GRSM | @ 1&JJS", 
    "@ 1&each",
]

ADD_COL_PATTERN_FUN = [
    pattern_for_substring,  # return 6; For each to group by
    pattern_for_age,        # return 7; age
    pattern_for_IN_TABLE,   # return 4; skip these pattern
    pattern_to_continue,    # return 1
    pattern_to_follow_last, # return 2
    pattern_to_new_where,   # return 3
    pattern_for_skip,       # return 5; unrecognized each
]

ADD_COL_PATTERNS = create_pattern_toks([], ADD_COL_PATTERNS)