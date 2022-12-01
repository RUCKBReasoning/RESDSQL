import json
import re

from sql_metadata import Parser
from torch.utils.data import Dataset

class ColumnAndTableRankerDataset(Dataset):
    def __init__(
        self,
        dir_: str = None,
        sample: dict = None,
        use_original_name: bool = False,
        use_contents: bool = True,
        use_column_type: bool = False,
        add_pk_info: bool = False,
        add_fk_info: bool = True,
    ):
        super(ColumnAndTableRankerDataset, self).__init__()

        self.questions: list[str] = []
        
        self.all_column_infos: list[list[list[str]]] = []
        self.all_column_labels: list[list[list[int]]] = []

        self.all_table_names: list[list[str]] = []
        self.all_table_labels: list[list[int]] = []
        self.fk_infos: list[list[list[int]]] = []

        if dir_ is None:
            if sample is None:
                raise ValueError("you must specify a dataset directory (i.e., param dir_) or a single sample.")
            dataset = [sample]
        else:
            with open(dir_, 'r', encoding = 'utf-8') as f:
                dataset = json.load(f)
        
        for data in dataset:
            column_names_in_one_db = []
            column_names_original_in_one_db = []
            extra_column_info_in_one_db = []
            column_labels_in_one_db = []

            table_names_in_one_db = []
            table_names_original_in_one_db = []
            table_labels_in_one_db = []

            for table_id in range(len(data["db_schema"])):
                column_names_original = data["db_schema"][table_id]["column_names_original"]
                table_name_original = data["db_schema"][table_id]["table_name_original"]
                column_names_original_in_one_db.append(column_names_original)
                table_names_original_in_one_db.append(table_name_original)

                table_name = data["db_schema"][table_id]["table_name_original"] if use_original_name else data["db_schema"][table_id]["table_name"]
                table_names_in_one_db.append(table_name)
                table_labels_in_one_db.append(data["table_labels"][table_id])

                column_names = data["db_schema"][table_id]["column_names_original"] if use_original_name else data["db_schema"][table_id]["column_names"] 
                column_names_in_one_db.append(column_names)
                column_labels_in_one_db += data["column_labels"][table_id]
                
                extra_column_info = ["" for _ in range(len(column_names))]

                if use_column_type:
                    column_types = data["db_schema"][table_id]["column_types"]
                    for column_id, column_type in enumerate(column_types):
                        extra_column_info[column_id] += column_type

                if use_contents:
                    column_contents = data["db_schema"][table_id]["db_contents"]
                    for column_id, content in enumerate(column_contents):
                        if len(content) != 0:
                            if len(extra_column_info[column_id]) != 0:
                                extra_column_info[column_id] +=  " , " + " , ".join(content)
                            else:
                                extra_column_info[column_id] += " , ".join(content)
                
                if add_pk_info:
                    primary_keys = data["pk"]
                    for column_id, column_name_original in enumerate(column_names_original):
                        column_is_pk = False
                        for pk in primary_keys:
                            if table_name_original == pk["table_name"] and column_name_original == pk["column_name"]:
                                column_is_pk = True
                        if column_is_pk:
                            if len(extra_column_info[column_id]) != 0:
                                extra_column_info[column_id] += " , [PK]"
                            else:
                                extra_column_info[column_id] += "[PK]"
                
                extra_column_info_in_one_db.append(extra_column_info)
            
            fk_info = []
            if add_fk_info:
                foreign_keys = data["fk"]
                for fk_id, fk in enumerate(foreign_keys):
                    source_table_name = fk["source_table_name"]
                    source_column_name = fk["source_column_name"]
                    target_table_name = fk["target_table_name"]
                    target_column_name = fk["target_column_name"]

                    source_table_id, target_table_id = 0, 0
                    for table_id, table_name in enumerate(table_names_original_in_one_db):
                        if source_table_name == table_name:
                            source_table_id = table_id
                        if target_table_name == table_name:
                            target_table_id = table_id
                    
                    source_column_id, target_column_id = 0, 0

                    for column_id, column_name_original in enumerate(column_names_original_in_one_db[source_table_id]):
                        if source_column_name == column_name_original:
                            source_column_id = column_id
                            if len(extra_column_info_in_one_db[source_table_id][column_id]) != 0:
                                if "[FK]" not in extra_column_info_in_one_db[source_table_id][column_id]:
                                    extra_column_info_in_one_db[source_table_id][column_id] += " , [FK]"
                            else:
                                extra_column_info_in_one_db[source_table_id][column_id] += "[FK]"
                    
                    for column_id, column_name_original in enumerate(column_names_original_in_one_db[target_table_id]):
                        if target_column_name == column_name_original:
                            target_column_id = column_id
                            if len(extra_column_info_in_one_db[target_table_id][column_id]) != 0:
                                if "[FK]" not in extra_column_info_in_one_db[target_table_id][column_id]:
                                    extra_column_info_in_one_db[target_table_id][column_id] += " , [FK]"
                            else:
                                extra_column_info_in_one_db[target_table_id][column_id] += "[FK]"
                    
                    fk_info.append([source_table_id, target_table_id, source_column_id, target_column_id])
            
            # column_info = column name + extra column info
            column_infos_in_one_db = []
            for table_id in range(len(table_names_in_one_db)):
                column_infos = []
                for column_name, extra_column_info in zip(column_names_in_one_db[table_id], extra_column_info_in_one_db[table_id]):
                    if len(extra_column_info) != 0:
                        column_infos.append(column_name + " ( " + extra_column_info + " ) ")
                    else:
                        column_infos.append(column_name)
                column_infos_in_one_db.append(column_infos)
            
            self.questions.append(data["question"])
            
            self.all_table_names.append(table_names_in_one_db)
            self.all_table_labels.append(table_labels_in_one_db)

            self.all_column_infos.append(column_infos_in_one_db)
            self.all_column_labels.append(column_labels_in_one_db)

            self.fk_infos.append(fk_info)
    
    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, index):
        question = self.questions[index]

        table_names_in_one_db = self.all_table_names[index]
        table_labels_in_one_db = self.all_table_labels[index]

        column_infos_in_one_db = self.all_column_infos[index]
        column_labels_in_one_db = self.all_column_labels[index]
        fk_info_in_one_db = self.fk_infos[index]

        return question, table_names_in_one_db, table_labels_in_one_db, column_infos_in_one_db, column_labels_in_one_db, fk_info_in_one_db

class Text2SQLDataset(Dataset):
    def __init__(
        self,
        dir_: str,
        use_contents: bool,
        add_fk_info: bool,
        add_pk_info: bool,
        mode: str,
        output_sql_skeleton: bool = False
    ):
        super(Text2SQLDataset).__init__()
        
        self.mode = mode

        self.input_sequences: list[str] = []
        self.output_sequences: list[str] = []
        self.db_ids: list[str] = []

        with open(dir_, 'r', encoding = 'utf-8') as f:
            dataset = json.load(f)
        
        for data in dataset:
            question = data["question"]
            db_schema = ""
            # db_schema = " | " + data["db_id"]
            required_table_names = []
            for table_id in range(len(data["db_schema"])):
                if table_id <= 4:
                    # add table name
                    db_schema += " | " + data["db_schema"][table_id]["table_name_original"] + " : "
                    required_table_names.append(data["db_schema"][table_id]["table_name_original"])

                    column_info_list = []
                    for column_id in range(len(data["db_schema"][table_id]["column_names_original"])):
                        if column_id <= 5:
                            # extract column name
                            column_name = data["db_schema"][table_id]["column_names_original"][column_id]
                            column_addition_info = ""
                            # extract db contents if them exists
                            if use_contents and len(data["db_schema"][table_id]["db_contents"][column_id]) != 0:
                                column_addition_info += " , ".join(data["db_schema"][table_id]["db_contents"][column_id])

                            if add_pk_info:
                                # determine if the current column is a primary key
                                current_column_is_pk = False
                                for pk_info in data["pk"]:
                                    if pk_info["table_name"] == data["db_schema"][table_id]["table_name_original"] \
                                        and pk_info["column_name"] == data["db_schema"][table_id]["column_names_original"][column_id]:
                                        current_column_is_pk = True
                                if current_column_is_pk:
                                    column_addition_info += "[PK]" if column_addition_info == "" else " , [PK]"
                            
                            column_info = column_name if column_addition_info == "" else column_name + " ( " + column_addition_info + " ) "
                            column_info_list.append(column_info)

                    # add column info (including column name and db contents)
                    db_schema += " , ".join(column_info_list)
            
            if add_fk_info:
                fk_infos = data["fk"]
                for source_table_name in required_table_names:
                    for target_table_name in required_table_names:
                        for fk_info in fk_infos:
                            if source_table_name == fk_info["source_table_name"] and target_table_name == fk_info["target_table_name"]:
                                db_schema += " | " + fk_info["source_table_name"] + "." + fk_info["source_column_name"] + " = " + fk_info["target_table_name"] + "." + fk_info["target_column_name"]
                    
            # remove additional spaces
            while "  " in db_schema:
                db_schema = db_schema.replace("  ", " ")
            
            self.input_sequences.append((question + db_schema).strip())
            self.db_ids.append(data["db_id"])
            if self.mode in ["train", "eval"]:
                target_sql = self.normalize_sql(data["query"]).strip()
                if output_sql_skeleton:
                    target_sql_skeleton = self.extract_sql_skeleton(target_sql)
                    self.output_sequences.append(target_sql_skeleton + " | " + target_sql)
                else:
                    self.output_sequences.append(target_sql)
            elif self.mode == "test":
                pass
            else:
                raise ValueError("Invalid mode. Please choose from 'train or eval'")
    
    def normalize_sql(self, sql):
        def white_space_fix(s):
            sql_tokens = Parser(s).tokens
            s = " ".join([token.value for token in sql_tokens])

            return s

        def lower(s):
            # Convert everything except text between single quotation marks to lower case
            in_quotation = False
            out_s = ""
            for char in s:
                if in_quotation:
                    out_s += char
                else:
                    out_s += char.lower()
                
                if char == "'":
                    if in_quotation:
                        in_quotation = False
                    else:
                        in_quotation = True
            
            return out_s

        def remove_semicolon(s):
            if s.endswith(";"):
                s = s[:-1]
            return s
        
        def double2single(s):
            return s.replace("\"", "'") 
        
        def add_asc(s):
            pattern = re.compile(r'order by (?:\w+ \( \S+ \)|\w+\.\w+|\w+)(?: (?:\+|\-|\<|\<\=|\>|\>\=) (?:\w+ \( \S+ \)|\w+\.\w+|\w+))*')
            if "order by" in s and "asc" not in s and "desc" not in s:
                for p_str in pattern.findall(s):
                    s = s.replace(p_str, p_str + " asc")

            return s

        def replace_table_alias_func(s):
            tables_aliases = Parser(s).tables_aliases
            new_tables_aliases = {}
            for i in range(1,11):
                if "t{}".format(i) in tables_aliases.keys():
                    new_tables_aliases["t{}".format(i)] = tables_aliases["t{}".format(i)]
            
            tables_aliases = new_tables_aliases
            for k, v in tables_aliases.items():
                s = s.replace("as " + k + " ", "")
                s = s.replace(k, v)
            
            return s

        while "  " in sql:
            sql = sql.replace("  ", " ")
        
        process_sql_func = lambda x : replace_table_alias_func(add_asc(white_space_fix(lower(double2single(remove_semicolon(x))))))
        
        return process_sql_func(sql)
    
    # mask table names, table aliases, column names, values and join operator in SQL
    def extract_sql_skeleton(self, sql):
        parsed_sql = Parser(sql)

        sql_tokens = []
        for token in parsed_sql.tokens:
            if token.is_keyword and token.value in ['length', 'language', 'result', 'location', 'share', 'type', 'year']:
                sql_tokens.append("_")
            else:
                sql_tokens.append(token.value)
        
        sql = " ".join(sql_tokens)
        
        parsed_sql = Parser(sql)
        table_names = parsed_sql.tables
        try:
            column_names = parsed_sql.columns
        except:
            return sql
        
        tables_aliases = parsed_sql.tables_aliases
        table_alias2table_name = {}
        for i in range(1,11):
            if "t{}".format(i) in tables_aliases.keys():
                table_alias2table_name["t{}".format(i)] = tables_aliases["t{}".format(i)]
        
        for k,v in table_alias2table_name.items():
            sql = sql.replace(k, v)

        parsed_sql = Parser(sql)
        
        sql_tokens = []
        for token in parsed_sql.tokens:
            # mask table names
            if token.value in table_names:
                sql_tokens.append("_")
            # mask column names
            elif token.value in column_names:
                sql_tokens.append("_")
            # mask "*" column
            elif token.value == "*":
                sql_tokens.append("_")
            # mask string values
            elif token.value.startswith("'") and token.value.endswith("'"):
                sql_tokens.append("_")
            # mask number
            elif token.value.isdigit():
                sql_tokens.append("_")
            else:
                sql_tokens.append(token.value)

        sql = " ".join(sql_tokens)

        # remove JOIN ON
        sql = sql.replace("on _ = _ and _ = _", "on _ = _")
        sql = sql.replace(" on _ = _", "")
        pattern3 = re.compile("_ as _ (?:join _ as _ ?)+")
        sql = re.sub(pattern3, "_ ", sql)
        pattern4 = re.compile("_ (?:join _ ?)+")
        sql = re.sub(pattern4, "_ ", sql)

        # remove AS
        sql = sql.replace("_ as _", "_")

        return sql


    def __len__(self):
        return len(self.input_sequences)
    
    def __getitem__(self, index):
        if self.mode == "train":
            return self.input_sequences[index], self.output_sequences[index], self.db_ids[index]
        elif self.mode in ['eval', "test"]:
            return self.input_sequences[index], self.db_ids[index]