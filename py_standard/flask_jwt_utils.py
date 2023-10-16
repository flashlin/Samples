import datetime
import os

from dotenv import load_dotenv
from flask import current_app, Flask
from flask import request, Blueprint, jsonify
from flask_cors import CORS, cross_origin
from flask_jwt_extended import create_access_token, jwt_required, get_jwt_identity, JWTManager

"""
app.register_blueprint(blueprint_jwt_login, url_prefix='/api/v1/auth')
"""


def create_auth_blueprint(app: Flask):
    blueprint_jwt_login = Blueprint('jwt_login_v1', __name__)
    app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET_KEY')
    jwt = JWTManager(app)

    @blueprint_jwt_login.route('/login', methods=['POST'])
    @cross_origin()
    def login():
        req = request.get_json()
        login_name = req['loginName']
        password = req['password']
        login_name = login_name.lower()

        user_service = current_app.config.get('UserService')
        if user_service is None:
            raise ValueError(f"Null flask config['UserService']")

        user = user_service.get_user(login_name)
        if not user_service.check_password(password, user):
            return jsonify({'message': 'Invalid username or password'}), 401

        token = create_access_token(identity=user.login_name,
                                    fresh=True,
                                    expires_delta=datetime.timedelta(minutes=10))
        return jsonify({'token': token}), 200

    @blueprint_jwt_login.route('/register', methods=['POST'])
    @cross_origin()
    def register():
        req = request.get_json()
        login_name = req['loginName']
        password = req['password']
        login_name = login_name.lower()

        user_service = current_app.config.get('UserService')
        if user_service is None:
            raise ValueError(f"Null flask config['UserService']")

        user = user_service.get_user(login_name)
        if user.login_name == login_name:
            return jsonify({'message': 'LoginName already registered'}), 406

        # TODO
        user_service.register()
        return jsonify({'token': token}), 200

    @blueprint_jwt_login.route('/refreshToken', methods=['GET'])
    @cross_origin()
    @jwt_required()
    def refresh_token():
        current_login_name = get_jwt_identity()
        token = create_access_token(identity=current_login_name,
                                    fresh=True,
                                    expires_delta=datetime.timedelta(minutes=10))
        return jsonify({'token': token}), 200

    return blueprint_jwt_login
