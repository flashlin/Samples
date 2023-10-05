import pymysql


class MysqlDbContext:
    def __init__(self):
        db_settings = {
            "host": "127.0.0.1",
            "port": 3306,
            "user": "flash",
            "password": "pass",
            "db": "gpt_db",
            "charset": "utf8",
        }
        self.conn = pymysql.connect(**db_settings)

    def execute(self, sql: str, args: tuple):
        conn = self.conn
        with conn.cursor() as cursor:
            # sql = f"INSERT INTO class(id,name) VALUES (%s,%s)"
            cursor.execute(sql, args)
            conn.commit()

    def query(self, sql: str, args: tuple=None):
        conn = self.conn
        results = []
        with conn.cursor() as cursor:
            # VALUES ('%s', '%s', '%d', '%c', '%d' )" % \
            #    ('Max', 'Su', 25, 'F', 2800)
            if args is None:
                cursor.execute(sql)
            else:
                cursor.execute(sql, args)
            columns = [column[0] for column in cursor.description]
            row = cursor.fetchone()
            obj = {}
            for i in range(len(row)):
                column = columns[i]
                setattr(obj, column, row[i])
            results += obj
            # results = cursor.fetchall()
            conn.commit()
        return results


if __name__ == '__main__':
    db = MysqlDbContext()
    results = db.query("select * from Customers")
    print(f"{results=}")