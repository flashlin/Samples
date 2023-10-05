from dataclasses import dataclass
from datetime import datetime


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


class GptRepo:
    def __init__(self):
        pass

    def get_user_conversation(self, conversation_id: int) -> Conversation:
        pass

    def create_conversation(self) -> ConversationMessage:
       pass

    def add_conversation_message(self, data: ConversationMessage):
        pass