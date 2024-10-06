from backend.backend_server.log import log as logger
from backend.backend_server.util.error_enum import BusinessEnum
from backend.backend_server.util.jwt_util import create_account, verify_account, logout_account


class UserController(object):

    def __init__(self, request, response):
        self.request = request
        self.response = response

    def login(self, username, password):
        logger.info("username: {0}, password {1}".format(username, password))
        assert username, "{0} | {1}".format(username, BusinessEnum.ERROR_ARGV)
        assert password, "{0} | {1}".format(password, BusinessEnum.ERROR_ARGV)
        token = self.request.cookies.get("token")
        if verify_account(token):
            self.response.set_cookie(key="token", value=token, httponly=True)
            return
        if username and password:
            pass

    def register(self):
        pass
        # insert

    def logout(self):
        token = self.request.cookies.get("token")
        logger.info("before token: {0}".format(token))
        assert verify_account(token), BusinessEnum.UN_LOGIN
        token = logout_account(token)
        logger.info("after logout token: {0}".format(token))
        self.response.set_cookie(key="token", value=token, httponly=True)
