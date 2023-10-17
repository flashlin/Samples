from dataclasses import dataclass
from gpt_repo_utils import GptRepo


@dataclass
class GetConversationMessagesReq:
    conversation_id: int
    login_name: str


class GptService:
    def __init__(self, gpt_db: GptRepo):
        self.gpt_db = gpt_db

    def get_conversation_messages(self, req: GetConversationMessagesReq):
