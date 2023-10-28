import os
from datetime import datetime, timezone
import pymysql
from repo_types import DbConfig
from logging_utils import logger
from obj_utils import dump, create_dataclass
from types import SimpleNamespace


class MysqlDbContext:
    def __init__(self, config: DbConfig):
        db_settings = {
            "charset": "utf8",
            "host": '127.0.0.1',
            "port": 3306,
            "db": 'gpt_db',
            "user": os.getenv("MYSQL_USER"),
            "password": os.getenv("MYSQL_PASSWORD")
        }
        db_settings.update(vars(config))
        self.conn = pymysql.connect(**db_settings)

    def execute(self, sql: str, args: tuple = None):
        conn = self.conn
        with conn.cursor() as cursor:
            # sql = f"INSERT INTO class(id,name) VALUES (%s,%s)"
            cursor.execute(sql, args)
            inserted_id_or_rowcount = inserted_id = cursor.lastrowid
            if inserted_id == 0:
                inserted_id_or_rowcount = cursor.rowcount
            conn.commit()
        return inserted_id_or_rowcount

    def query(self, sql: str, args: tuple = None) -> list[dict]:
        """
        :param sql: "select * from Customers where id = %d or name = %s or ch = %c
        :param args: (1, "flash", 'c')
        :return:
        """
        logger.info(f"MysqlDbContext::query {sql=} {args=}")
        conn = self.conn
        with conn.cursor() as cursor:
            if args is None:
                cursor.execute(sql)
            else:
                cursor.execute(sql, args)
            columns = [column[0] for column in cursor.description]
            results = []
            row = cursor.fetchone()
            while row is not None:
                # obj = type("DynamicEntity", (), {})()
                # for i in range(len(row)):
                #     column = columns[i]
                #     setattr(obj, column, row[i])
                data_dict = dict(zip(columns, row))
                results.append(data_dict)
                row = cursor.fetchone()
            # results = cursor.fetchall()
            # logger.info(f"{results=}")
            conn.commit()
        return results

    def query_objects(self, sql: str, args: tuple = None) -> list[object]:
        result = self.query(sql, args)
        object_list = []
        for row in result:
            obj = SimpleNamespace(**row)
            object_list.append(obj)
        return object_list

    def query_dataclass(self, data_class_type, sql: str, args: tuple = None) -> list[object]:
        result = self.query(sql, args)
        data_obj_list = []
        for row in result:
            data_obj = create_dataclass(data_class_type, **row)
            data_obj_list.append(data_obj)
        return data_obj_list


def to_utc_time_str(time: datetime = None) -> str:
    if time is None:
        time = datetime.now(timezone.utc)
    return time.strftime('%Y-%m-%d %H:%M:%S')


def utc_time() -> datetime:
    return datetime.now(timezone.utc)


def test():
    db = MysqlDbContext()
    results = db.query("select * from Customers")
    print(dump(results))
    results = db.query("select * from Customers where loginName=%s", "flash1")
    print(dump(results))
