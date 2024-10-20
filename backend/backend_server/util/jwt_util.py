import time
import jwt
from backend.backend_server.config import JWT_KEY
from backend.backend_server.database import UserCRUD
from backend.backend_server.log import log as logger
from backend.backend_server.util.error_enum import BusinessEnum

__all__ = ["create_account", "verify_account"]


def _jwt2value(item_jwt: str):
    if not item_jwt:
        return {}
    decoded: dict = jwt.decode(item_jwt, JWT_KEY, algorithms=["HS256"])
    logger.info("jwt {0} -> decoded value {1}".format(item_jwt, decoded))
    return decoded


def create_account(item_dict: dict):
    assert item_dict, "item_dict {0}".format(BusinessEnum.ERROR_ARGV)
    assert item_dict["username"], "item_dict.username {0}".format(BusinessEnum.ERROR_ARGV)
    item_dict["gen_time"] = time.time()
    seven_days_seconds = 7 * 24 * 3600
    item_dict["expiration_time"] = time.time() + seven_days_seconds
    encoded = jwt.encode(item_dict, JWT_KEY, algorithm="HS256")
    logger.info("values {0} -> gen jwt {1}".format(item_dict, encoded))
    return encoded


async def verify_account(str_encoded_token: str):
    logger.info("token {0}".format(str_encoded_token))
    token_value = _jwt2value(str_encoded_token)
    cur_time = time.time()
    if not token_value or not token_value.get("expiration_time") or token_value.get(
            "expiration_time", 0) < cur_time or not token_value.get("username"):
        return False
    user_info = await UserCRUD.get_item_user(token_value.get("username"))
    if user_info:
        return token_value.get("username")
    else:
        return False


async def verify_username_password(username, password: str):
    user_info = await UserCRUD.get_item_user(username)
    if user_info:
        if user_info.get("password") == password:
            return True
    return False


def logout_account(str_encoded_token: str):
    token_value = _jwt2value(str_encoded_token)
    cur_time = time.time()
    token_value["expiration_time"] = cur_time - 1
    return jwt.encode(token_value, JWT_KEY, algorithm="HS256")
