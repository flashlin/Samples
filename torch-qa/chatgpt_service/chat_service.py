from dataclasses import dataclass
from datetime import datetime

from conversation import Conversation, ChatMessage


class ChatService:
    def __init__(self):
        self.conv = Conversation()

    def message(self, req: ChatMessage):
        conv = self.conv
        if req.conversation_id == -1:
            username = ""
            conversation_id = conv.new_conversation(username)
            print(f"{conversation_id=}")


if __name__ == '__main__':
    chat_service = ChatService()
    chat_service.message(ChatMessage(
        conversation_id=-1,
        message='Please use c# write Hello'
    ))
