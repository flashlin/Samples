from dataclasses import dataclass
from datetime import datetime
from abc import ABC, abstractmethod


@dataclass
class DbConfig:
    host: str
    port: int
    user: str
    password: str
    db: str


@dataclass
class Conversation:
    conversation_id: int
    login_name: str
    create_on: datetime


@dataclass
class ConversationMessage:
    conversation_id: int
    role_name: str
    message: str
    create_on: datetime


@dataclass
class AddConversationReq:
    conversation_id: int
    login_name: str


class GptRepo(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_user_conversation(self, conversation_id: int) -> Conversation:
        pass

    @abstractmethod
    def create_conversation(self, login_name: str) -> ConversationMessage:
       pass

    @abstractmethod
    def add_conversation_message(self, req: AddConversationReq):
        pass
