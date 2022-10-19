import pandas as pd
import psycopg2 as pg2
import psycopg2.extras

class Database:
    def __init__(self):
        # self.host = "localhost"
        # self.db = "chatbot_db"
        # self.username = "chatbot_db"
        # self.password = "root"
        # self.port = 5433
        self.host = "20.194.252.15"
        self.db = "chatbot_db"
        self.username = "root"
        self.password = "root"
        self.port = 5432
        self.cur = None
        self.conn = None

    def connect(self):
        self.conn = pg2.connect(database=self.db, user=self.username, password=self.password, port=self.port,
                                host=self.host)
        self.cur = self.conn.cursor()

    def execute_query(self, query, value):
        self.connect()
        self.cur.execute(query, value)
        self.conn.commit()

    def close(self):
        self.cur.close()
        self.conn.close()

    def postgresql_to_dataframe(self, select_query, column_names):
        self.connect()
        try:
            self.cur.execute(select_query)
        except (Exception, pg2.DatabaseError) as error:
            print("Error: %s" % error)
            self.close()
            return 1

        tupples = self.cur.fetchall()
        self.close()

        # We just need to turn it into a pandas dataframe
        df = pd.DataFrame(tupples, columns=column_names)
        return df

    def getCategory(self):
        return self.postgresql_to_dataframe("select * from category", ["category_id", "category_name"])

    def getAllQuestion(self):
        return self.postgresql_to_dataframe("select * from tb_question", ["question_id","category_id","group_id","question_content"])

    def getAllAnswer(self):
        return self.postgresql_to_dataframe("select * from tb_answer", ["answer_id","category_id", "answer_content"])

    def insert(self, table_name, list_field, list_value):
        pg2.extras.register_uuid()
        len_value = "VALUES("
        for x in range(len(list_field)-1):
            len_value+="%s,"
        len_value +="%s)";
        sql = f"INSERT INTO {table_name}({','.join(list_field)}) {len_value}"
        value = tuple(list_value)
        self.execute_query(sql, value)

    def getAnswerById(self,id):
        self.connect()
        try:
            self.cur.execute(f"select answer_content from answer where qa_id ='{id}'")
        except (Exception, pg2.DatabaseError) as error:
            print("Error: %s" % error)
            self.close()
            return 1
        tupples = self.cur.fetchall()
        return tupples[0]

    def getModel(self):
        self.connect()
        try:
            self.cur.execute("select * from tb_model")
        except (Exception, pg2.DatabaseError) as error:
            print("Error: %s" % error)
            self.close()
            return 1
        tupples = self.cur.fetchall()
        return tupples[0]
