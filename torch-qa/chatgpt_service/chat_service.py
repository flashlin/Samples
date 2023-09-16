from dataclasses import dataclass
from datetime import datetime

from chatgpt_service.chat_db import SqliteRepo, ChatMessageEntity


@dataclass
class ConversationMessage:
    role_name: str
    message: str


@dataclass
class ChatMessage:
    conversationId: int
    roleName: str
    message: str
    createOn: datetime


@dataclass
class ConversationStack:
    conversation_id: int
    message_stack: [ChatMessage]


class Conversation:
    def __init__(self):
        self.chat_db = SqliteRepo()

    def add_message(self, req: ConversationMessage):
        self.chat_db.add_message(ChatMessageEntity(
            conversationId=1,
            roleName=req.role_name,
            messageText=req.message,
            createOn=datetime.now()
        ))

    def get_conversation(self, conversation_id):
        messages = self.chat_db.get_messages(conversation_id)
        message_stack = []
        for message in messages:
            message_stack.append(ChatMessage(roleName=message.roleName, message=message.messageText))
        return ConversationStack(conversation_id, message_stack)


class ChatService:
    def __init__(self):
        self.chat_db = SqliteRepo()

    def message(self, req: ChatMessage):
        pass