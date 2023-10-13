from datetime import datetime

from sqlalchemy import create_engine, or_, Column, Integer, String, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base

from obj_utils import dump


Base = declarative_base()


class Customer(Base):
    __tablename__ = 'Customers'
    Id = Column(Integer, primary_key=True, autoincrement=True)
    LoginName = Column(String(255))
    CreateOn = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

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
            customer = session.query(Customer).filter(
                or_(Customer.LoginName == 'flash', Customer.LoginName == 'flash2')).first()
            customer.LoginName = "flash2"
            # session.delete(customer)
            session.commit()


def test():
    db2 = MysqlDbContext2()
    print("---")
    db2.query()
    db2.update()
    db2.query()



