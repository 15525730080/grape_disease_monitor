import traceback
from inspect import trace

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from backend.backend_server.video_control.video_handle import video_image_handler
from backend.backend_server.log import log
router = APIRouter()
@router.websocket("/video_ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_bytes()
            video_image_handler.process_image(data)
            await websocket.send_text("Image received successfully")
    except BaseException as e:
        log.error(traceback.print_exc())