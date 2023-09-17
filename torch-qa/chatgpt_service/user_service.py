from dataclasses import dataclass
from datetime import datetime
from chat_db import SqliteRepo
from flask_bcrypt import Bcrypt
from app import app


mybcrypt = Bcrypt(app)

@dataclass
class User:
    username: str
    password_hash: str
    is_enabled: bool
    createOn: datetime

    def check_password(self, password):
        if not self.is_enabled:
            return False
        return mybcrypt.check_password_hash(self.password_hash, password)


class UserService:
    def __init__(self, db: SqliteRepo):
        self.db = db

    def get_user(self, username):
        user = self.db.get_user(username)
        return User(
            username=user.username,
            password_hash=user.password,
            is_enabled=user.isEnabled,
            createOn=user.createOn
        )

    def create_user(self, username: str, password: str) -> bool:
        password_hash = mybcrypt.generate_password_hash(password).decode('utf-8')
        return self.db.create_user(username, password_hash)
