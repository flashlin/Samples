from dataclasses import dataclass
from datetime import datetime
from abc import ABC, abstractmethod


@dataclass
class CreateUserReq:
    login_name: str
    password_hash: str


@dataclass
class CreateUserResp:
    is_success: bool
    error_message: str


@dataclass
class ConversationEntity:
    Id: int
    LoginName: str
    CreateOn: datetime


@dataclass
class ConversationMessageEntity:
    Id: int
    ConversationsId: int
    RoleName: str
    Message: str
    CreateOn: datetime


@dataclass
class AddConversationMessageReq:
    ConversationId: int
    RoleName: str
    Message: str
    CreateOn: datetime


@dataclass
class AddConversationReq:
    conversation_id: int
    login_name: str
    message: str


@dataclass
class CustomerEntity:
    Id: int
    LoginName: str
    Password: str
    CreateOn: datetime


class GptRepo(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def create_user(self, req: CreateUserReq) -> CreateUserResp:
        pass

    @abstractmethod
    def get_user(self, username: str) -> CustomerEntity:
        pass

    @abstractmethod
    def get_user_conversation(self, conversation_id: int) -> ConversationEntity:
        pass

    @abstractmethod
    def get_last_conversation(self, login_name: str) -> ConversationEntity:
       pass

    @abstractmethod
    def create_conversation(self, login_name: str) -> ConversationEntity:
       pass

    @abstractmethod
    def add_conversation_message(self, req: AddConversationReq) -> None:
        pass

    @abstractmethod
    def get_conversation_message_list(self, conversation_id: int) -> list[ConversationMessageEntity]:
        pass
