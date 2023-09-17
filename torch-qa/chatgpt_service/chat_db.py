import sqlite3
from datetime import datetime
from dataclasses import dataclass

users_ddl = '''CREATE TABLE users (
    username varchar(50) NOT NULL primary key,
    password varchar(255),
    isEnabled BOOLEAN,
    createOn DATETIME
)
'''

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


@dataclass
class UserEntity:
    username: str
    password: str
    isEnabled: bool
    createOn: datetime

    @staticmethod
    def empty():
        return UserEntity(
            username="",
            password="",
            isEnabled=False,
            createOn=datetime.min
        )


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
        self.execute(users_ddl)
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
                     "VALUES (:conversationId, :roleName, :messageText, DateTime('now'))",
                     vars(message))

    def get_messages(self, conversation_id: int):
        sql = "SELECT id, roleName, messageText FROM ("
        sql += ("SELECT id, roleName, messageText from chatMessages "
                "where conversationId = :conversationId order by id DESC LIMIT 100")
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

    def get_user(self, username: str) -> UserEntity:
        rows = self.query("select username, password, isEnabled, createOn from users where username = :username",
                          {
                              'username': username
                          })
        if len(rows) <= 0:
            return UserEntity.empty()

        row = rows[0]
        return UserEntity(
            username=row['username'],
            password=row['password'],
            isEnabled=row['isEnabled'],
            createOn=row['createOn']
        )

    def create_user(self, username: str, password_hash: str):
        count = self.execute("INSERT INTO users(username, password, isEnabled, createOn) VALUES(:username, :password, 1, DateTime('now'))",
                     {
                         'username': username,
                         'password': password_hash
                     })
        return count != 0
