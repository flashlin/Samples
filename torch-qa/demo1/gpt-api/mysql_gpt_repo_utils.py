import os
from datetime import datetime, timezone
from data_utils import hash_password
from gpt_repo_utils import GptRepo, CreateUserReq, CreateUserResp, Conversation, ConversationMessage, \
    AddConversationReq, CustomerEntity
from obj_utils import dump
from repo_types import DbConfig
from mysql_utils import MysqlDbContext, to_utc_time_str
from dotenv import load_dotenv


class MysqlGptRepo(GptRepo):
    def __init__(self, config: DbConfig):
        super().__init__()
        self.db = MysqlDbContext(config)

    def create_user(self, req: CreateUserReq) -> CreateUserResp:
        old_users = self.db.query('SELECT Id, LoginName, CreateOn FROM Customers WHERE LoginName=%s LIMIT 1',
                                  (req.login_name,))
        if len(old_users) != 0:
            return CreateUserResp(
                is_success=False,
                error_message=f'{req.login_name} is already exists.'
            )

        self.db.execute('INSERT INTO Customers(LoginName, Password, CreateOn) VALUES(%s, %s, %s)',
                        (req.login_name, hash_password(req.password), to_utc_time_str()))

        return CreateUserResp(
            is_success=True,
            error_message=f''
        )

    def get_user(self, login_name: str) -> CustomerEntity:
        old_users = self.db.query('SELECT Id, LoginName, CreateOn FROM Customers WHERE LoginName=%s LIMIT 1',
                                  (login_name,))
        if len(old_users) == 0:
            return CustomerEntity(
                Id=0,
                LoginName='',
                Password='',
                CreateOn=datetime.min
            )
        return old_users[0]

    def get_user_conversation(self, conversation_id: int) -> Conversation:
        results = self.db.query('select Id, LoginName, CreateOn from Conversations where Id=%d LIMIT 1',
                                (conversation_id,))
        if len(results) == 0:
            return Conversation(conversation_id=-1,
                                login_name='',
                                create_on=datetime.min)
        item = results[0]
        return Conversation(conversation_id=item.Id,
                            login_name=item.LoginName,
                            create_on=item.CreateOn)

    def create_conversation(self, login_name: str) -> ConversationMessage:
        inserted_id = self.db.execute("INSERT INTO Conversations(LoginName, CreateOn) VALUES(%s, NOW())",
                                      (login_name,))

        conversation = ConversationMessage(
            conversation_id=inserted_id,
            role_name="System",
            message="",
            create_on=datetime.now(timezone.utc)
        )

        self.db.execute("INSERT INTO ConversationsDetail(ConversationsId, RoleName, Message, CreateOn) "
                        "VALUES(%s, %s, %s, %s)",
                        (conversation.conversation_id,
                         conversation.role_name,
                         conversation.message,
                         conversation.create_on.strftime('%Y-%m-%d %H:%M:%S')))

        return conversation

    def add_conversation_message(self, req: AddConversationReq):
        results = self.db.query("SELECT ConversationsId, LoginName FROM Conversations WHERE ConversationsId=%d",
                                (req.conversation_id,))
        conversation = results[0]
        if conversation.LoginName != req.login_name:
            raise Exception(f"ConversationsId {req.conversation_id} is not {req.login_name}'s message")


def test():
    load_dotenv()
    gpt = MysqlGptRepo(DbConfig(
        host='127.0.0.1',
        port=3306,
        db='gpt_db',
        user=os.getenv("MYSQL_USER"),
        password=os.getenv("MYSQL_PASSWORD")
    ))

    resp = gpt.create_user(CreateUserReq(
        login_name='flash',
        password='pass'
    ))
    print(f"create user {resp=}")

    user = gpt.get_user('flash')
    print(f"{dump(user)=}")

    conversation = gpt.create_conversation('flash')
    print(f"{conversation=}")


if __name__ == '__main__':
    test()
