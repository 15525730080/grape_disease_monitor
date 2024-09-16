from fastapi import FastAPI
from backend.backend_server.video_control.video_handle import video_image_handler
from backend_server.router import router

app = FastAPI()
app.include_router(router)

if __name__ == "__main__":
    import uvicorn

    video_image_handler.start_custom_thread()
    uvicorn.run(app, host="0.0.0.0", port=8000)
