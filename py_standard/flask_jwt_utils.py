import datetime
from flask import current_app
from flask import request, Blueprint, jsonify
from flask_cors import CORS, cross_origin
from flask_jwt_extended import create_access_token, jwt_required, get_jwt_identity

"""
app.register_blueprint(blueprint_jwt_login, url_prefix='/api/v1/auth')
"""
blueprint_jwt_login = Blueprint('jwt_login_v1', __name__)


@blueprint_jwt_login.route('/login')
@cross_origin()
def login():
    req = request.get_json()
    username = req['username']
    password = req['password']
    username = username.lower()

    user_service = current_app.config.get('UserService')
    if user_service is None:
        raise ValueError(f"Null flask config['UserService']")

    user = user_service.get_user(username)
    if not user.check_password(password):
        return jsonify({'message': 'Invalid username or password'}), 401

    token = create_access_token(identity=user.username,
                                fresh=True,
                                expires_delta=datetime.timedelta(minutes=10))
    return jsonify({'token': token}), 200


@blueprint_jwt_login.route('/refreshToken')
@cross_origin()
@jwt_required()
def refresh_token():
    current_user = get_jwt_identity()
    token = create_access_token(identity=current_user,
                                fresh=True,
                                expires_delta=datetime.timedelta(minutes=10))
    return jsonify({'token': token}), 200
