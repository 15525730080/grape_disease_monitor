import traceback
from fastapi import APIRouter, WebSocket, Request, Response, Depends
from starlette.responses import RedirectResponse

from backend.backend_server.user_control.view import UserController
from backend.backend_server.video_control.video_handle import video_image_handler
from backend.backend_server.log import log as logger
from backend.backend_server.param_entity import User

router = APIRouter()


# 依赖函数
def get_user_controller(request: Request, response: Response):
    return UserController(request, response)


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
def login(user: User, controller: UserController = Depends(get_user_controller)):
    return controller.login(user.username, user.password)
