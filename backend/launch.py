import sys

sys.path.insert(0, r"e:\\postgraduatecode\\grape_disease_monitor")
from fastapi import FastAPI
from backend.backend_server.video_control.video_handle import video_image_handler
from backend_server.router import router, CustomMiddleware

app = FastAPI()
app.add_middleware(CustomMiddleware)
app.include_router(router)


if __name__ == "__main__":
    import uvicorn

    video_image_handler.start_custom_thread()
    uvicorn.run("launch:app", host="0.0.0.0", port=8000, reload=True)
