from dataclasses import dataclass
from datetime import datetime
from flask import Flask, request, render_template, jsonify, redirect, url_for

from chat_db import SqliteRepo
from conversation import Conversation, ChatMessage, ConversationMessage, ConversationStack
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


class EmptyLLM(LLM):
    def message(self, message_stack: ConversationStack) -> str:
        return ""

    def response_api(self, message: dict):
        return json.dumps(message)


class ChatService:
    llm: LLM = EmptyLLM()

    def __init__(self, llm: LLM = None):
        self.conv = Conversation()
        if llm is not None:
            self.llm = llm

    def message(self, req: ChatMessage):
        conv = self.conv
        if req.conversation_id == -1:
            username = ""
            conversation_id = conv.new_conversation(username)
            conv.add_message(ConversationMessage(
                conversation_id=conversation_id,
                role_name='system',
                message=system_prompt
            ))
        else:
            conversation_id = req.conversation_id

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


if __name__ == '__main__':
    chat_service = ChatService()
    chat_service.message(ChatMessage(
        conversation_id=-1,
        role_name='user',
        message='Please use c# write Hello'
    ))
