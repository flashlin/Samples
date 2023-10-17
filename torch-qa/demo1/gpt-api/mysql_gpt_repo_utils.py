import os
from datetime import datetime, timezone
from data_utils import hash_password
from gpt_repo_utils import GptRepo, CreateUserReq, CreateUserResp, Conversation, ConversationMessage, \
    AddConversationReq, CustomerEntity
from llama2_utils import DEFAULT_SYSTEM_PROMPT
from obj_utils import dump
from repo_types import DbConfig
from mysql_utils import MysqlDbContext, to_utc_time_str
from dotenv import load_dotenv


class MysqlGptRepo(GptRepo):
    def __init__(self, config: DbConfig = None):
        super().__init__()
        if config is None:
            config = DbConfig(
                host='127.0.0.1',
                port=3306,
                db='gpt_db',
                user=os.getenv("MYSQL_USER"),
                password=os.getenv("MYSQL_PASSWORD")
            )
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
                        (req.login_name, req.password_hash, to_utc_time_str()))

        return CreateUserResp(
            is_success=True,
            error_message=f''
        )

    def get_user(self, login_name: str) -> CustomerEntity:
        old_users = self.db.query('SELECT Id, LoginName, Password, CreateOn FROM Customers WHERE LoginName=%s LIMIT 1',
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

    def get_last_conversation(self, login_name: str) -> Conversation:
        results = self.db.query('select Id, LoginName, CreateOn from Conversations where LoginName=%s ORDER BY Id DESC LIMIT 1',
                                (login_name,))
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
            role_name="system",
            message=DEFAULT_SYSTEM_PROMPT,
            create_on=datetime.now(timezone.utc)
        )
        self.add_conversation_detail(conversation)
        return conversation

    def add_conversation_message(self, req: AddConversationReq):
        results = self.db.query("SELECT Id, LoginName FROM Conversations WHERE Id=%s LIMIT 1",
                                (req.conversation_id,))
        conversation = results[0]
        if conversation.LoginName != req.login_name:
            raise Exception(f"ConversationsId {req.conversation_id} is not {req.login_name}'s message")

        last_messages = self.db.query("SELECT RoleName FROM ConversationsDetail WHERE ConversationsId=%s ORDER BY Id DESC LIMIT 1",
                      (req.conversation_id,))

        if last_messages[0].RoleName != 'assistant' and last_messages[0].RoleName != 'system':
            raise Exception(f"ConversationsId {req.conversation_id} The replay message unexpectedly interrupts.")

        self.add_conversation_detail(
            ConversationMessage(
                conversation_id=req.conversation_id,
                role_name='user',
                message=req.message,
                create_on=datetime.now(timezone.utc)
            )
        )

    def get_conversation_message_list(self, conversation_id: int) -> [ConversationMessage]:
        results = self.db.query("SELECT Id, ConversationsId, RoleName, Message, CreateOn FROM ConversationsDetail "
                               "WHERE ConversationsId=%s ORDER BY Id DESC LIMIT 100", (conversation_id,))
        ordered_result = results[::-1]

        result = self.db.query("SELECT Id, ConversationsId, RoleName, Message, CreateOn FROM ConversationsDetail "
                               "WHERE ConversationsId=%s ORDER BY Id LIMIT 1", (conversation_id,))
        ordered_result[0] = result[0]
        return ordered_result

    def add_conversation_detail(self, data: ConversationMessage):
        self.db.execute("INSERT INTO ConversationsDetail(ConversationsId, RoleName, Message, CreateOn) "
                        "VALUES(%s, %s, %s, %s)",
                        (data.conversation_id,
                         data.role_name,
                         data.message,
                         to_utc_time_str(data.create_on),)
                        )


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
        password_hash='pass'
    ))
    print(f"create user {resp=}")

    user = gpt.get_user('flash')
    print(f"{dump(user)=}")

    conversation = gpt.create_conversation('flash')
    print(f"{conversation=}")

    gpt.add_conversation_message(AddConversationReq(
        conversation_id=4,
        login_name='flash',
        message='Hello'
    ))

    last_messages = gpt.get_conversation_message_list(4)
    print(f"{dump(last_messages)=}")


if __name__ == '__main__':
    test()
