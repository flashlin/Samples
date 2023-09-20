import datetime
from app import app
from flask import Flask, request, render_template, jsonify, Response
from flask_jwt_extended import JWTManager
from flask_jwt_extended import create_access_token
from flask_jwt_extended import jwt_required
from flask_jwt_extended import get_jwt_identity
from flask_cors import CORS, cross_origin

from chat_db import SqliteRepo
from chat_service import ChatService, UserChatMessage
from user_service import UserService

cors = CORS(app, resources={r"/api/*": {"origins": "*"}})
app.config['CORS_HEADERS'] = 'Content-Type'
app.config['JWT_SECRET_KEY'] = 'my-secret-key'
jwt = JWTManager(app)
chat_service = ChatService()


@app.route('/')
@app.route('/index')
def home():
    print('index')
    template = render_template('index.html')
    return template


@app.route("/api/login", methods=['POST'])
@cross_origin()
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


@app.route("/api/refresh", methods=['POST'])
@cross_origin()
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


@app.route("/api/register", methods=['POST'])
@cross_origin()
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


@app.route("/api/message", methods=['POST'])
@cross_origin()
@jwt_required()
def message():
    current_user = get_jwt_identity()
    req = request.get_json()
    conversation_id = req['conversationId']
    user_message = req['message']
    response_message = chat_service.message(UserChatMessage(
        username=current_user,
        message=user_message,
        conversation_id=conversation_id,
        role_name='user'
    ))
    return jsonify({'message': response_message})


def stream(current_user: str, conversation_id: int, user_message: str):
    response_message = chat_service.message_stream(UserChatMessage(
        username=current_user,
        message=user_message,
        conversation_id=conversation_id,
        role_name='user'
    ))
    return response_message


@app.route('/api/message_stream', methods=['POST'])
@cross_origin()
@jwt_required()
def message_stream():
    current_user = get_jwt_identity()
    req = request.get_json()
    conversation_id = req['conversationId']
    user_message = req['message']
    return Response(stream(current_user, conversation_id, user_message), mimetype='text/event-stream')


if __name__ == "__main__":
    app.run(debug=True)
