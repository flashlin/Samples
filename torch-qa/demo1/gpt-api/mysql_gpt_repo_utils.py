import os
from datetime import datetime, timezone
from types import SimpleNamespace

from data_utils import hash_password
from gpt_repo_utils import GptRepo, CreateUserReq, CreateUserResp, ConversationEntity, ConversationMessageEntity, \
    AddConversationReq, CustomerEntity, AddConversationMessageReq
from llama2_utils import DEFAULT_SYSTEM_PROMPT
from obj_utils import dump, dict_to_obj, create_dataclass
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

        user = old_users[0]
        return create_dataclass(CustomerEntity, **user)

    def get_user_conversation(self, conversation_id: int) -> ConversationEntity:
        results = self.db.query('select Id, LoginName, CreateOn from Conversations where Id=%s LIMIT 1',
                                (conversation_id,))
        if len(results) == 0:
            return ConversationEntity(Id=-1,
                                      LoginName='',
                                      CreateOn=datetime.min)
        item = results[0]
        return create_dataclass(ConversationEntity, **item)

    def get_last_conversation(self, login_name: str) -> ConversationEntity:
        results = self.db.query(
            'select Id, LoginName, CreateOn from Conversations where LoginName=%s ORDER BY Id DESC LIMIT 1',
            (login_name,))
        if len(results) == 0:
            return ConversationEntity(Id=-1,
                                      LoginName='',
                                      CreateOn=datetime.min)
        item = results[0]
        return create_dataclass(ConversationEntity, **item)

    def create_conversation(self, login_name: str) -> ConversationEntity:
        inserted_id = self.db.execute("INSERT INTO Conversations(LoginName, CreateOn) VALUES(%s, NOW())",
                                      (login_name,))

        self.add_conversation_detail(AddConversationMessageReq(
            ConversationId=inserted_id,
            RoleName="system",
            Message=DEFAULT_SYSTEM_PROMPT,
            CreateOn=datetime.now(timezone.utc)
        ))

        return self.get_user_conversation(inserted_id)

    def add_user_conversation_message(self, req: AddConversationReq):
        results = self.db.query("SELECT Id, LoginName FROM Conversations WHERE Id=%s LIMIT 1",
                                (req.conversation_id,))
        if len(results) == 0:
            raise Exception(f"ConversationsId {req.conversation_id} not found in Conversations")

        conversation = SimpleNamespace(**results[0])
        if conversation.LoginName != req.login_name:
            raise Exception(f"ConversationsId {req.conversation_id} is not {req.login_name}'s message")

        last_messages = self.db.query(
            "SELECT RoleName FROM ConversationsDetail WHERE ConversationsId=%s ORDER BY Id DESC LIMIT 1",
            (req.conversation_id,))

        last = SimpleNamespace(**last_messages[0])
        if last.RoleName != 'assistant' and last.RoleName != 'system':
            raise Exception(f"ConversationsId {req.conversation_id} The replay message unexpectedly interrupts.")

        self.add_conversation_detail(
            AddConversationMessageReq(
                ConversationId=req.conversation_id,
                RoleName='user',
                Message=req.message,
                CreateOn=datetime.now(timezone.utc)
            )
        )

    def get_conversation_message_list(self, conversation_id: int) -> [ConversationMessageEntity]:
        results = self.db.query("SELECT Id, ConversationsId, RoleName, Message, CreateOn FROM ConversationsDetail "
                                "WHERE ConversationsId=%s ORDER BY Id DESC LIMIT 100", (conversation_id,))
        ordered_result = results[::-1]

        result = self.db.query("SELECT Id, ConversationsId, RoleName, Message, CreateOn FROM ConversationsDetail "
                               "WHERE ConversationsId=%s ORDER BY Id LIMIT 1", (conversation_id,))
        ordered_result[0] = result[0]
        ordered_result = [create_dataclass(ConversationMessageEntity, **item) for item in ordered_result]
        return ordered_result

    def get_last_conversation_message(self, conversation_id: int) -> ConversationMessageEntity:
        result = self.db.query("SELECT Id, ConversationsId, RoleName, Message, CreateOn FROM ConversationsDetail "
                                "WHERE ConversationsId=%s ORDER BY Id DESC LIMIT 1", (conversation_id,))
        if len(result) == 0:
            return ConversationMessageEntity(
                Id=-1,
                ConversationsId=conversation_id,
                RoleName="",
                Message="",
                CreateOn=datetime.now(timezone.utc)
            )
        return create_dataclass(ConversationMessageEntity, **result[0])

    def delete_conversation_message(self, conversation_id: int) -> None:
        self.db.execute("DELETE ConversationsDetail WHERE Id=%s", (conversation_id,))

    def add_conversation_detail(self, data: AddConversationMessageReq):
        self.db.execute("INSERT INTO ConversationsDetail(ConversationsId, RoleName, Message, CreateOn) "
                        "VALUES(%s, %s, %s, %s)",
                        (data.ConversationId,
                         data.RoleName,
                         data.Message,
                         to_utc_time_str(data.CreateOn),)
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

    gpt.add_user_conversation_message(AddConversationReq(
        conversation_id=conversation.Id,
        login_name='flash',
        message='Hello'
    ))

    last_messages = gpt.get_conversation_message_list(4)
    print(f"{dump(last_messages)=}")


if __name__ == '__main__':
    test()
