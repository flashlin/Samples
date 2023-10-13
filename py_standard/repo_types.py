from dataclasses import dataclass


@dataclass
class DbConfig:
    host: str
    port: int
    user: str
    password: str
    db: str
