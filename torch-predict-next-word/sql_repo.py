
import sqlite3

class SqlRepo:
    def __init__(self):
        self.conn = sqlite3.connect('./output/sql.db')

    def execute(self, sql, parameters_tuple=None):
        c = self.conn.cursor()
        if parameters_tuple is not None:
            c.execute(sql, parameters_tuple)
        else:
            c.execute(sql)
        c.close()
    
    def query(self, sql, parameters_tuple=None):
        """
        sql = "select id from customer where name=?"
        parameters_tuple = (name,)
        """
        c = self.conn.cursor()
        if parameters_tuple is not None:
            c.execute(sql, parameters_tuple)
        else:
            c.execute(sql)
        results = c.fetchall()
        c.close()
        return results
