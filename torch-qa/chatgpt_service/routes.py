from app import app
from flask import Flask, request, render_template, jsonify
from flask_jwt_extended import JWTManager
from chat_db import SqliteRepo
from chat_service import ChatService
from user_service import UserService

app.config['JWT_SECRET_KEY'] = 'my-secret-key'
jwt = JWTManager(app)

chat_db = SqliteRepo()
user_service = UserService(chat_db)
chat_service = ChatService(chat_db)

@app.route('/')
@app.route('/index')
def home():
    print('index')
    template = render_template('index.html')
    return template


@app.route("/login", methods=['POST'])
def login():
    username = request.form['username']
    password = request.form['password']
    username = username.lower()
    user = user_service.get_user(username)
    if not user.check_password(password):
        return jsonify({'message': 'Invalid username or password'}), 401
    token = jwt.encode({'user_id': user.id}, app.config['JWT_SECRET_KEY'])
    return jsonify({'token': token}), 200


@app.route("/register", methods=['POST'])
def register():
    username = request.form['username']
    password = request.form['password']
    username = username.lower()
    created = user_service.create_user(username, password)
    if not created:
        return jsonify({'message': 'user not created'})
    return jsonify({'message': ''})


if __name__ == "__main__":
    app.run(debug=True)
