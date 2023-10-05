import pymysql


class MysqlDbContext:
    def __init__(self):
        db_settings = {
            "host": "127.0.0.1",
            "port": 3306,
            "user": "flash",
            "password": "pass",
            "db": "demo",
            "charset": "utf8",
        }
        self.conn = pymysql.connect(**db_settings)

    def execute(self, sql):
        conn = self.conn
        with conn.cursor() as cursor:
            sql = f"INSERT INTO class(id,name) VALUES (%s,%s)"
            cursor.execute(sql, ("1", "science"))
            conn.commit()

    def query(self, sql: str, args: tuple):
        conn = self.conn
        with conn.cursor() as cursor:
            # VALUES ('%s', '%s', '%d', '%c', '%d' )" % \
            #    ('Max', 'Su', 25, 'F', 2800)
            sql = f"INSERT INTO class(id,name) VALUES (%s,%s)"
            cursor.execute(sql, args)
            columns = [column[0] for column in cursor.description]
            data = cursor.fetchone()
            results = cursor.fetchall()
            conn.commit()
        return results
