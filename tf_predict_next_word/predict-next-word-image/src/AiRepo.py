
import pyodbc


class AiRepo:
    conn = pyodbc.connect(
        'DRIVER={ODBC Driver 17 for SQL Server};SERVER=localhost,4331;DATABASE=QueryDB;UID=sa;PWD=Passw0rd!;')

    def __init__(self, logger):
        self.logger = logger
        self.init()

    def init(self):
        self.logger.info('checking database')
        # has_table = self.is_table_exists('Sentences')
        # if not has_table:
        #     self.logger.info('create table')
        #     self.execute_sql("CREATE TABLE Sentences("
        #                      "  [ID] INT primary key IDENTITY(1,1) NOT NULL,"
        #                      "  [Sentence] NVARCHAR(1000) NOT NULL,"
        #                      "  [CreateOn] DATETIME NOT NULL"
        #                      ")")

    def is_table_exists(self, table_name):
        sql = f"SELECT object_id FROM sys.tables WHERE name='{table_name}' AND SCHEMA_NAME(schema_id)='dbo';"
        rows = self.query_sql(sql)
        has_data = bool(rows)
        return has_data

    def execute_sql(self, sql):
        conn = self.conn
        cursor = conn.cursor()
        cursor.execute(sql)
        cursor.close()

    def query_sql(self, sql):
        conn = self.conn
        cursor = conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()
        cursor.close()
        return rows

    def get_sql_history_list(self):
        rows = self.query_sql("SELECT TOP 100 [SqlCode] FROM [SqlHistory] WITH(NOLOCK) ORDER BY [CreatedOn] DESC")
        for row in rows:
            self.logger.info(f'{row.SqlCode}')
        return rows
