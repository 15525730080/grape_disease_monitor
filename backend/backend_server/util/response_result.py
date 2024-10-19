from typing import Any


class ResponseResult(dict):
    def __init__(self, code: int, data: dict[Any, Any] = dict(), message: str = ''):
        super().__init__()
        self.code = code
        self.data = data if data is not None else {}
        self.message = message
        self.update({
            'code': self.code,
            'data': self.data,
            'message': self.message
        })
