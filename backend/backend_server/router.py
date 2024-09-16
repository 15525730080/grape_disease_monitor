import traceback
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from backend.backend_server.video_control.video_handle import video_image_handler
from backend.backend_server.log import log as logger
router = APIRouter()
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


