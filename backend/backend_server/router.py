import traceback
from fastapi import APIRouter, WebSocket, Request, Response, Depends
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import RedirectResponse, JSONResponse
from backend.backend_server.user_control.view import UserController
from backend.backend_server.util.response_result import ResponseResult
from backend.backend_server.video_control.video_handle import video_image_handler
from backend.backend_server.log import log as logger
from backend.backend_server.param_entity import User

router = APIRouter()


# 依赖函数
def get_user_controller(request: Request, response: Response):
    return UserController(request, response)


class CustomMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        try:
            response = await call_next(request)
        except BaseException as e:
            logger.error(traceback.format_exc())
            return JSONResponse(content=ResponseResult(code=500, message=str(e)), status_code=500)
        return response


@router.websocket("/video_ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_bytes()
            logger.info("received")
            video_image_handler.process_image(data)
            await websocket.send_text("Image received successfully")
    except BaseException as e:
        logger.error(traceback.print_exc())


@router.get("/")
def index():
    return RedirectResponse(url="/static/index.html")


@router.post("/login")
async def login(user: User, controller: UserController = Depends(get_user_controller)):
    return await controller.login(user.username, user.password)


@router.post("/register")
async def register(user: User, controller: UserController = Depends(get_user_controller)):
    return await controller.register(user.name, user.username, user.password)


@router.post("/logout")
async def register(user: User, controller: UserController = Depends(get_user_controller)):
    return await controller.register(user.name, user.username, user.password)
