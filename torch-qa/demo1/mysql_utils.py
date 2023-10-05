from typing import Tuple
from datetime import datetime
import pymysql
import json
from sqlalchemy import create_engine, Column, Integer, String, Sequence, DateTime, or_
# from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, declarative_base

from demo1.gpt_repo_utils import GptRepo, Conversation, DbConfig, ConversationMessage
from obj_utils import dump_obj, dump


class MysqlDbContext:
    def __init__(self, config: DbConfig):
        db_settings = {
            "charset": "utf8",
        }
        db_settings.update(vars(config))
        self.conn = pymysql.connect(**db_settings)

    def execute(self, sql: str, args: tuple):
        conn = self.conn
        with conn.cursor() as cursor:
            # sql = f"INSERT INTO class(id,name) VALUES (%s,%s)"
            cursor.execute(sql, args)
            conn.commit()

    def query(self, sql: str, args: Tuple[...] = None) -> list[any]:
        """
        :param sql: "select * from Customers where id = %d or name = %s or ch = %c
        :param args: (1, "flash", 'c')
        :return:
        """
        conn = self.conn
        results = []
        with conn.cursor() as cursor:
            if args is None:
                cursor.execute(sql)
            else:
                cursor.execute(sql, args)
            columns = [column[0] for column in cursor.description]
            row = cursor.fetchone()
            while row is not None:
                obj = type("DynamicEntity", (), {})()
                for i in range(len(row)):
                    column = columns[i]
                    setattr(obj, column, row[i])
                results.append(obj)
                row = cursor.fetchone()
            # results = cursor.fetchall()
            conn.commit()
        return results


class MysqlGptRepo(GptRepo):
    def __init__(self, config: DbConfig):
        super().__init__()
        self.db = MysqlDbContext(config)

    def get_user_conversation(self, conversation_id: int) -> Conversation:
        db = self.db
        results = db.query('select Id, LoginName, CreateOn from Conversations where Id=%d LIMIT 1',
                           (conversation_id, ))
        if len(results) == 0:
            return Conversation(conversation_id=-1,
                                login_name='',
                                create_on=datetime.min)
        item = results[0]
        return Conversation(conversation_id=item.Id,
                            login_name=item.LoginName,
                            create_on=item.CreateOn)

    def create_conversation(self) -> ConversationMessage:
        pass

    def add_conversation_message(self, data: ConversationMessage):
        pass


class MysqlDbContext2:
    def __init__(self):
        DATABASE_URL = 'mysql+pymysql://flash:pass@localhost:3306/gpt_db'
        self.engine = create_engine(DATABASE_URL, echo=False)

    def query(self):
        engine = self.engine
        Session = sessionmaker(bind=engine)
        session = Session()
        # customers = session.query(Customer).all()
        customers = (session.query(Customer)
                     .filter(or_(Customer.LoginName == 'flash', Customer.LoginName == 'flash2'))
                     .all())
        for customer in customers:
            print(f"{dump(customer)}")

    def update(self):
        Session = sessionmaker(bind=self.engine)
        session = Session()
        customer = session.query(Customer).filter_by(LoginName='flash').first()
        if customer is not None:
            customer = session.query(Customer).filter(or_(Customer.LoginName=='flash',Customer.LoginName=='flash2')).first()
            customer.LoginName = "flash2"
            # session.delete(customer)
            session.commit()


Base = declarative_base()


class Customer(Base):
    __tablename__ = 'Customers'
    Id = Column(Integer, primary_key=True, autoincrement=True)
    LoginName = Column(String(255))
    CreateOn = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


if __name__ == '__main__':
    db = MysqlDbContext()
    results = db.query("select * from Customers")
    print(dump(results))
    results = db.query("select * from Customers where loginName=%s", "flash1")
    print(dump(results))
    db2 = MysqlDbContext2()
    print("---")
    db2.query()
    db2.update()
    db2.query()
