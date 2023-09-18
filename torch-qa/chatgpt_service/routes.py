import datetime

from app import app
from flask import Flask, request, render_template, jsonify, Response
from flask_jwt_extended import JWTManager
from flask_jwt_extended import create_access_token
from flask_jwt_extended import jwt_required
from flask_jwt_extended import get_jwt_identity
from chat_db import SqliteRepo
from chat_service import ChatService
from user_service import UserService

app.config['JWT_SECRET_KEY'] = 'my-secret-key'
jwt = JWTManager(app)

@app.route('/')
@app.route('/index')
def home():
    print('index')
    template = render_template('index.html')
    return template


@app.route("/login", methods=['POST'])
def login():
    req = request.get_json()
    username = req['username']
    password = req['password']
    username = username.lower()
    user_service = UserService()
    user = user_service.get_user(username)
    if not user.check_password(password):
        return jsonify({'message': 'Invalid username or password'}), 401
    # datetime.timedelta(minutes=15)
    token = create_access_token(identity=user.username,
                                fresh=True,
                                expires_delta=datetime.timedelta(minutes=10))
    return jsonify({'token': token}), 200


@app.route("/refresh", methods=['POST'])
@jwt_required()
def refresh_token():
    current_user = get_jwt_identity()
    token = create_access_token(identity=current_user,
                                fresh=True,
                                expires_delta=datetime.timedelta(minutes=10))
    return jsonify({
        'token': token,
        'msg': ''
        }), 200


@app.route("/register", methods=['POST'])
def register():
    req = request.get_json()
    username = req['username']
    password = req['password']
    username = username.lower()
    user_service = UserService()
    created = user_service.create_user(username, password)
    if not created:
        return jsonify({'message': 'user not created'})
    return jsonify({'message': ''})


@app.route("/message", methods=['POST'])
@jwt_required()
def message():
    req = request.get_json()
    current_user = get_jwt_identity()
    conversation_id = req['conversationId']
    message = req['message']
    return jsonify({'message': current_user})


def stream(conversation_id: int, input_text: str):
    current_user = get_jwt_identity()
    chat_service = ChatService()



@app.route('/completion', methods=['GET', 'POST'])
@jwt_required()
def completion_api():
    req = request.get_json()
    if request.method == "POST":
        conversation_id = req['conversationId']
        input_text = req['message']
        return Response(stream(conversation_id, input_text), mimetype='text/event-stream')
    return Response(None, mimetype='text/event-stream')


if __name__ == "__main__":
    app.run(debug=True)
