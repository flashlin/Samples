from dataclasses import dataclass
from datetime import datetime
from flask import Flask, request, render_template, jsonify, redirect, url_for

from conversation import Conversation, ChatMessage, ConversationMessage

system_prompt = "You are an AI assistant that talks like a pirate in rhyming couplets."

class ChatService:
    def __init__(self):
        self.conv = Conversation()

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

        messages = conv.add_message(ConversationMessage(
            conversation_id=req.conversation_id,
            role_name='user',
            message=system_prompt
        ))

        # chat = chat_model.ChatCompletion.create(
        #     model="gpt-3.5-turbo-0301",
        #     messages=messages
        # )
        # response = chat['choices'][0]['message']['content']
        response = ""
        conv.add_message(ConversationMessage(
            conversation_id=req.conversation_id,
            role_name='assistant',
            message=response
        ))
        data = {"id": req.conversation_id, "response": response}
        return jsonify(data)


if __name__ == '__main__':
    chat_service = ChatService()
    chat_service.message(ChatMessage(
        conversation_id=-1,
        message='Please use c# write Hello'
    ))
