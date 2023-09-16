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
messages_ddl = '''
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


class SqliteRepo:
    def __init__(self, db_file='data/chatgpt.db'):
        self.conn = sqlite3.connect('data/chatgpt.db')

    def close(self):
        self.conn.close()

    def execute(self, sql, parameters):
        cursor = self.conn.cursor()
        cursor.execute(sql, parameters)
        cursor.close()

    def query(self, sql, parameters):
        cursor = self.conn.cursor()
        cursor.execute(sql, parameters)
        rows = cursor.fetchall()
        cursor.close()
        return rows

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
