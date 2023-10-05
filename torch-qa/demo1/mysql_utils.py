from datetime import datetime

import pymysql
import json
from sqlalchemy import create_engine, Column, Integer, String, Sequence, DateTime, or_
# from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, declarative_base

from obj_utils import dump_obj, dump


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

    def query(self, sql: str, args: tuple = None):
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
