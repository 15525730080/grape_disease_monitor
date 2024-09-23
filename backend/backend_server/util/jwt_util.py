import time

import jwt

from backend.backend_server.config import JWT_KEY
from backend.backend_server.log import log as logger


def gen_jwt(item_dict: dict):
    assert item_dict, "非法参数 {0}".format(item_dict)
    item_dict["gen_time"] = time.time()
    seven_days_seconds = 7 * 24 * 3600
    item_dict["expiration_time"] = time.time() + seven_days_seconds
    encoded = jwt.encode(item_dict, JWT_KEY, algorithm="HS256")
    logger.info("values {0} -> gen jwt {1}".format(item_dict, encoded))
    return encoded


def jwt2value(item_jwt: dict):
    decoded = jwt.decode(item_jwt, JWT_KEY, algorithms=["HS256"])
    logger.info("jwt {0} -> decoded value {1}".format(item_jwt, decoded))


def verify_account(item_dict: dict):
    token_value = gen_jwt(item_dict)
    cur_time = time.time()
    if not token_value or not token_value.get("expiration_time") or token_value.get("expiration_time") > cur_time:
        return False
    return True
