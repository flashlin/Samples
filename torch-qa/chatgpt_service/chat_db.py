import sqlite3
from datetime import datetime
from dataclasses import dataclass

conversations_ddl = '''
CREATE TABLE conversations (
    id INTEGER primary key autoincrement,
    userName VARCHAR(50),
    summary VARCHAR(500),
    createOn DATETIME
)
'''

# message id comes from front end and represents order in message array
chat_messages_ddl = '''
CREATE TABLE chatMessages (
    id INTEGER primary key autoincrement,
    conversationId INTEGER,
    roleName VARCHAR(50),
    messageText VARCHAR(2000),
    createOn DATETIME
)
'''

@dataclass
class ChatMessageEntity:
    conversationId: int
    roleName: str
    messageText: str
    createOn: datetime


def dict_to_obj(d):
  row_obj = type('row', (object,), d)
  return row_obj()


class SqliteRepo:
    def __init__(self, db_file='data/chatgpt.db'):
        self.conn = sqlite3.connect('data/chatgpt.db')
        self.create_database()

    def close(self):
        self.conn.close()

    def create_database(self):
        if self.is_table_exists("conversations"):
            return
        self.execute(conversations_ddl)
        self.execute(chat_messages_ddl)

    def is_table_exists(self, table_name: str):
        table_info = self.query(f"PRAGMA table_info({table_name})")
        return len(table_info) > 0

    def execute(self, sql: str, parameters: dict = None) -> int:
        cursor = self.conn.cursor()
        if parameters is None:
            cursor.execute(sql)
        else:
            cursor.execute(sql, parameters)
        cursor.close()
        self.conn.commit()
        return cursor.lastrowid

    def query(self, sql, parameters: dict = None):
        cursor = self.conn.cursor()
        # self.conn.row_factory = sqlite3.Row
        if parameters is None:
            cursor.execute(sql)
        else:
            cursor.execute(sql, parameters)
        names = [description[0] for description in cursor.description]
        rows = cursor.fetchall()
        cursor.close()
        result = []
        for row in rows:
            item = {}
            for value, name in zip(row, names):
                item[name] = value
            # result.append(dict_to_obj(item))
            result.append(item)
        return result

    def add_message(self, message: ChatMessageEntity):
        self.execute("INSERT INTO chatMessages(conversationId, roleName, messageText, createOn) "
                     "VALUES (:conversationId, :roleName, :messageText, :createOn)",
                     message)

    def get_messages(self, conversation_id: int):
        sql = "SELECT roleName, messageText FROM ("
        sql += ("SELECT roleName, messageText from chatMessages "
                "where conversationId = :convertsationId order by id DESC LIMIT 100")
        sql += ") ORDER BY id ASC"
        rows = self.query(sql,
                          {
                              'conversationId': conversation_id
                          })
        return rows

    def add_conversation(self, username):
        last_rowid = self.execute("INSERT INTO conversations(userName, createOn) VALUES(:userName, DateTime('now'))", {
            'userName': username
        })
        rows = self.query("SELECT id FROM conversations WHERE rowid = :rowid", {
            'rowid': last_rowid
        })
        return rows[0]['id']
