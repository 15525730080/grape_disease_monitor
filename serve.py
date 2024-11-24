import os
from io import BytesIO

from fastapi import FastAPI, File, UploadFile
import base64
from predict.launch_ensemble_classifier import ensemble_predict

# 初始化 FastAPI 应用
app = FastAPI()

@app.get("/keep-live/")
async def keep_live():
    return 200


@app.post("/upload/")
async def upload_image(file: UploadFile):
    """
    上传图片并返回 JSON 响应。
    图片数据被存储在内存中而不是磁盘。
    """
    try:
        # 将图片文件读入内存
        image_data: bytes = await file.read()

        predicted_class, probabilities, custom_time = ensemble_predict(BytesIO(image_data))
        print(predicted_class, probabilities, custom_time)
        # 返回 JSON 响应
        return {
            "code": 200,
            "filename": file.filename,
            "predicted_class": predicted_class,
            "probabilities": probabilities,
            "custom_time": custom_time
        }
    except Exception as e:
        return {"code": 500, "msg": str(e)}


if __name__ == "__main__":
    import uvicorn

    # 启动服务，监听端口 202230
    uvicorn.run(app, host="0.0.0.0", port=20223)
