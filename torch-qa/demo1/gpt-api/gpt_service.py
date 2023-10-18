from dataclasses import dataclass
from datetime import datetime

from gpt_repo_utils import GptRepo, ConversationMessageEntity, AddConversationReq
from llama2_utils import GptMessage
from obj_utils import create_dataclass


@dataclass
class GetConversationMessagesReq:
    conversation_id: int
    login_name: str


@dataclass
class GetConversationMessagesResp:
    conversationId: int
    messages: list[GptMessage]


class GptService:
    def __init__(self, gpt_db: GptRepo):
        self.gpt_db = gpt_db

    def get_conversation_last_messages(self, login_name: str) -> GetConversationMessagesResp:
        gpt_db = self.gpt_db
        conversation = gpt_db.get_last_conversation(login_name)
        if conversation.Id == -1:
            conversation = gpt_db.create_conversation(login_name)
        result = self.convert_conversation_message_entity_to_gpt_message_list(gpt_db.get_conversation_message_list(conversation.Id))
        return GetConversationMessagesResp(
            conversationId=conversation.Id,
            messages=result
        )

    def cancel_last_conversation(self, login_name: str) -> bool:
        gpt_db = self.gpt_db
        conversation = gpt_db.get_last_conversation(login_name)
        if conversation.Id == -1:
            return False
        last_message = gpt_db.get_last_conversation_message(conversation.Id)
        if last_message.RoleName != 'user':
            return False
        gpt_db.delete_conversation_message(last_message.Id)
        return True

    def get_conversation_messages(self, req: GetConversationMessagesReq) -> list[GptMessage]:
        gpt_db = self.gpt_db
        conversation = gpt_db.get_user_conversation(req.conversation_id)
        if conversation.LoginName != req.login_name:
            return []
        data_list = gpt_db.get_conversation_message_list(req.conversation_id)
        return self.convert_conversation_message_entity_to_gpt_message_list(data_list)

    def convert_conversation_message_entity_to_gpt_message_list(self, data_list: list[ConversationMessageEntity]):
        result = []
        for item in data_list:
            row = GptMessage(id=item.Id,
                             role=item.RoleName,
                             content=item.Message,
                             create_on=item.CreateOn)
            result.append(row)
        return result

    def add_conversation_message(self, req: AddConversationReq):
        return self.gpt_db.add_user_conversation_message(req)
