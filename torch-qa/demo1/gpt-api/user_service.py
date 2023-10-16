from dataclasses import dataclass
from datetime import datetime
from gpt_repo_utils import GptRepo, CreateUserReq, CreateUserResp

from flask_bcrypt import Bcrypt


@dataclass
class User:
    login_name: str
    password_hash: str
    createOn: datetime


class UserService:
    def __init__(self, app, db: GptRepo):
        self.crypt = Bcrypt(app)
        self.db = db

    def get_user(self, username) -> User:
        user = self.db.get_user(username)
        return User(
            login_name=user.LoginName,
            password_hash=user.Password,
            createOn=user.CreateOn
        )

    def create_user(self, username: str, password: str) -> CreateUserResp:
        password_hash = self.crypt.generate_password_hash(password).decode('utf-8')
        resp = self.db.create_user(CreateUserReq(
            login_name=username,
            password=password_hash,
        ))
        return resp

    def check_password(self, password: str, user: User):
        if user.password_hash == '':
            return False
        return self.crypt.check_password_hash(user.password_hash, password)
