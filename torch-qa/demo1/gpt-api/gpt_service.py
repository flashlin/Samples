from dataclasses import dataclass
from gpt_repo_utils import GptRepo, ConversationMessageEntity, AddConversationReq


@dataclass
class GetConversationMessagesReq:
    conversation_id: int
    login_name: str


class GptService:
    def __init__(self, gpt_db: GptRepo):
        self.gpt_db = gpt_db

    def get_conversation_last_messages(self, login_name: str) -> [ConversationMessageEntity]:
        gpt_db = self.gpt_db
        conversation = gpt_db.get_last_conversation(login_name)
        if conversation.Id == -1:
            conversation = gpt_db.create_conversation(login_name)
        return gpt_db.get_conversation_message_list(conversation.Id)

    def get_conversation_messages(self, req: GetConversationMessagesReq) -> [ConversationMessageEntity]:
        gpt_db = self.gpt_db
        conversation = gpt_db.get_user_conversation(req.conversation_id)
        if conversation.LoginName != req.login_name:
            return []
        return gpt_db.get_conversation_message_list(req.conversation_id)

    def add_conversation_message(self, req: AddConversationReq):
        return self.gpt_db.add_conversation_message(req)