from dataclasses import dataclass
from datetime import datetime
from flask import Flask, request, render_template, jsonify, redirect, url_for

from chat_db import SqliteRepo
from conversation import Conversation, ConversationMessage, ConversationStack
from abc import ABC, abstractmethod
import json

system_prompt = "You are an AI assistant that talks like a pirate in rhyming couplets."


class LLM(ABC):
    @abstractmethod
    def message(self, message_stack: ConversationStack) -> str:
        pass

    @abstractmethod
    def response_api(self, message: dict):
        pass


@dataclass
class UserChatMessage:
    conversation_id: int
    username: str
    role_name: str
    message: str


class EmptyLLM(LLM):
    def message(self, message_stack: ConversationStack) -> str:
        return ""

    def response_api(self, message: dict):
        return json.dumps(message)


class ChatException(Exception):
    def __init__(self, message="A custom exception occurred"):
        self.message = message
        super().__init__(self.message)


class ChatService:
    llm: LLM = EmptyLLM()

    def __init__(self, llm: LLM = None):
        self.convo = Conversation()
        if llm is not None:
            self.llm = llm

    def message(self, req: UserChatMessage):
        conv = self.convo
        conversation_id = self.confirm_conversation_id(req)

        messages = conv.add_message(ConversationMessage(
            conversation_id=conversation_id,
            role_name='user',
            message=system_prompt
        ))

        # chat = chat_model.ChatCompletion.create(
        #     model="gpt-3.5-turbo-0301",
        #     messages=messages
        # )
        # response = chat['choices'][0]['message']['content']
        response = self.llm.message(messages)

        conv.add_message(ConversationMessage(
            conversation_id=conversation_id,
            role_name='assistant',
            message=response
        ))
        data = {"id": conversation_id, "response": response}
        return self.llm.response_api(data)

    def confirm_conversation_id(self, req: UserChatMessage):
        convo = self.convo
        if req.conversation_id == -1:
            conversation_id = convo.new_conversation(req.username)
            convo.add_message(ConversationMessage(
                conversation_id=conversation_id,
                role_name='system',
                message=system_prompt
            ))
        else:
            conversation_id = req.conversation_id
            if not convo.is_conversation_id_exists(req.conversation_id):
                raise ChatException(f"{conversation_id=} not found")
        return conversation_id

    def message_stream(self, req: UserChatMessage):
        conv = self.convo
        conversation_id = self.confirm_conversation_id(req)
        messages = conv.add_message(ConversationMessage(
            conversation_id=conversation_id,
            role_name='user',
            message=system_prompt
        ))

        # completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[
        #     {"role": "system", "content": "You're an assistant."},
        #     {"role": "user", "content": f"{prompt(input_text)}"},
        # ], stream=True, max_tokens=500, temperature=0)
        completion = self.llm.message(messages)
        for line in completion:
            if 'content' in line['choices'][0]['delta']:
                yield line['choices'][0]['delta']['content']



