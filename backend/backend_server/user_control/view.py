from backend.backend_server.database import UserCRUD
from backend.backend_server.log import log as logger
from backend.backend_server.util.error_enum import BusinessEnum
from backend.backend_server.util.jwt_util import create_account, verify_account, logout_account, \
    verify_username_password
from backend.backend_server.util.response_result import ResponseResult


class UserController(object):

    def __init__(self, request, response):
        self.request = request
        self.response = response

    async def login(self, username, password):
        logger.info("username: {0}, password: {1}".format(username, password))
        assert username, "{0} | {1}".format(username, BusinessEnum.ERROR_ARGV)
        assert password, "{0} | {1}".format(password, BusinessEnum.ERROR_ARGV)
        token = self.request.cookies.get("token")
        # 验证是否已登录
        if token and await verify_account(token):
            self.response.set_cookie(key="token", value=token, httponly=True)
            return ResponseResult(code=200, message="登录成功")
        if username and password:
            if await verify_username_password(username, password):
                token = create_account(dict(username=username))
                self.response.set_cookie(key="token", value=token, httponly=True)
                return ResponseResult(code=200, message="登录成功")
        return ResponseResult(code=500, message="登录失败")

    async def register(self, name, username, password):
        logger.info("username: {0}, password: {1}".format(username, password))
        assert username, "{0} | {1}".format(username, BusinessEnum.ERROR_ARGV)
        assert password, "{0} | {1}".format(password, BusinessEnum.ERROR_ARGV)
        assert not await UserCRUD.get_item_user(username=username), BusinessEnum.USER_EXISTS
        res = await UserCRUD.insert_item_user(name=name, username=username, password=password)
        if res:
            return ResponseResult(code=200, message="注册成功")

    async def logout(self):
        token = self.request.cookies.get("token")
        logger.info("before token: {0}".format(token))
        assert await verify_account(token), BusinessEnum.UN_LOGIN
        token = logout_account(token)
        logger.info("after logout token: {0}".format(token))
        self.response.set_cookie(key="token", value=token, httponly=True)
        return ResponseResult(code=200, message="退出成功")
