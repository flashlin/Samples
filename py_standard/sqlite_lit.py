from contextlib import closing
from typing import Mapping, Any, Sequence
import sqlite3

from obj_utils import dict_list_to_object_list


class SqliteMemDbContext:
    def __init__(self):
        self.conn = sqlite3.connect(':memory:')

    def execute(self, sql: str, args: tuple = None):
        with closing(self.conn.cursor()) as cursor:
            if args is None:
                cursor.execute(sql)
            else:
                cursor.execute(sql, args)

    def query(self, sql: str, args: tuple = None) -> list[dict]:
        with closing(self.conn.cursor()) as cursor:
            if args is None:
                rows = cursor.execute(sql).fetchall()
            else:
                rows = cursor.execute(sql, args).fetchall()
            results = []
            if rows is not None:
                columns = [description[0] for description in cursor.description]
                for row in rows:
                    data_dict = dict(zip(columns, row))
                    results.append(data_dict)
        return results

    def query_objects(self, sql: str, args: tuple = None) -> list[Any]:
        result = self.query(sql, args)
        return dict_list_to_object_list(result)


def test_sqlite_mem_db():
    db = SqliteMemDbContext()
    db.execute("""
    CREATE TABLE IF NOT EXISTS ConversationMessages (
                    id INTEGER PRIMARY KEY,
                    conversation_id INTEGER,
                    role_name TEXT,
                    content TEXT
                )
    """)
    db.execute("""INSERT INTO ConversationMessages (conversation_id, role_name, content) VALUES (?, ?, ?)
    """, (1, 'user', 'HELLO'))

    rows = db.query("""SELECT conversation_id, role_name, content FROM ConversationMessages""")
    print(f"{rows=}")

    rows = db.query_objects("""SELECT conversation_id, role_name, content FROM ConversationMessages""")
    print(f"{rows[0].role_name=}")