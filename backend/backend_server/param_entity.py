# 定义请求体模型
from pydantic import BaseModel


class User(BaseModel):
    username: str
    password: str