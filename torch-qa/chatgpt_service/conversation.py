from dataclasses import dataclass
from datetime import datetime

from chat_db import SqliteRepo, ChatMessageEntity


@dataclass
class ConversationMessage:
    conversation_id: int
    role_name: str
    message: str


@dataclass
class ConversationStack:
    conversation_id: int
    message_stack: [ConversationMessage]


class Conversation:
    def __init__(self):
        self.chat_db = SqliteRepo()

    def add_message(self, req: ConversationMessage):
        self.chat_db.add_message(ChatMessageEntity(
            conversationId=req.conversation_id,
            roleName=req.role_name,
            messageText=req.message,
            createOn=datetime.now()
        ))
        return self.get_conversation(req.conversation_id)

    def get_conversation(self, conversation_id):
        messages = self.chat_db.get_messages(conversation_id)
        message_stack = []
        for message in messages:
            message_stack.append(
                ConversationMessage(
                    conversation_id=conversation_id,
                    role_name=message['roleName'],
                    message=message['messageText'],
                )
            )
        return ConversationStack(conversation_id, message_stack)

    def new_conversation(self, username):
        conversation_id = self.chat_db.add_conversation(username)
        return conversation_id

    def is_conversation_id_exists(self, conversation_id):
        entity = self.chat_db.get_conversation(conversation_id)
        return entity.id != 0

