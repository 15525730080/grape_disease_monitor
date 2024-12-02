import base64
import time
import traceback
from io import BytesIO

from flask import Flask, request, jsonify
from flask_cors import CORS
from predict.img_diff import ImageComparator
from predict.launch_ensemble_classifier import ensemble_predict

from tinydb import TinyDB, Query
from tinydb.storages import JSONStorage
from tinydb.middlewares import CachingMiddleware
from filelock import FileLock  # 用于进程间文件锁

# 初始化 Flask 应用
app = Flask(__name__)
CORS(app)  # 启用跨域

# 初始化 TinyDB
db_path = "data.json"
db_lock = FileLock("data.json.lock")  # 防止进程间竞争
db = TinyDB(db_path, storage=CachingMiddleware(JSONStorage), ensure_ascii=False)


def diff_before_img(img):
    """
    比较当前图片和数据库中存储的图片，并决定是否更新图片。
    """
    with db_lock:  # 确保进程间安全
        db_table = db.table("before_img")
        record = db_table.get(Query().key == "before_img")
        if record:
            if time.time() - record["time"] > 60 * 30:
                db_table.upsert({"key": "before_img", "img": img, "time": time.time()}, Query().key == "before_img")
            else:
                try:
                    is_similar, similarity_score = ImageComparator.compare_images(bytes(record["img"]), img,
                                                                                  similarity_threshold=0.8)
                    if is_similar:
                        print(f"The images are similar with a similarity score of {similarity_score:.2f}.")
                        return True
                    else:
                        print(f"The images are not similar. Similarity score: {similarity_score:.2f}.")
                        db_table.upsert({"key": "before_img", "img": img, "time": time.time()},
                                        Query().key == "before_img")
                        return False
                except Exception as e:
                    print(f"Error comparing images: {e}")
                    db_table.upsert({"key": "before_img", "img": img, "time": time.time()}, Query().key == "before_img")
                    return False
        else:
            db_table.insert({"key": "before_img", "img": img, "time": time.time()})
            return False


@app.route("/keep-live/", methods=["GET"])
def keep_live():
    return "OK", 200


@app.route("/upload/", methods=["POST"])
def upload_image():
    """
    上传图片并返回 JSON 响应。
    图片数据被存储在内存中而不是磁盘。
    """
    try:
        # 获取上传的文件
        file = request.files.get("file")
        if not file:
            return jsonify({"code": 400, "msg": "No file provided"}), 400
        # 将图片文件读入内存
        image_data = file.read()
        image_data = bytes(image_data)
        diff_before_img(image_data)
        predicted_class, probabilities, custom_time = ensemble_predict(BytesIO(image_data))

        # 返回 JSON 响应
        return jsonify({
            "code": 200,
            "predicted_class": predicted_class,
            "probabilities": probabilities,
            "custom_time": custom_time
        })
    except Exception as e:
        return jsonify({"code": 500, "msg": str(e)}), 500


@app.route('/upload_base64/', methods=['POST'])
def upload_base64():
    try:
        data = request.get_json()
        base64_image = data.get("image", "")
        # 提取 Base64 部分
        header, encoded = base64_image.split(",", 1)
        image_data = base64.b64decode(encoded)
        image_data = bytes(image_data)
        diff_before_img(image_data)
        predicted_class, probabilities, custom_time = ensemble_predict(BytesIO(image_data))
        # 返回 JSON 响应
        return jsonify({
            "code": 200,
            "predicted_class": predicted_class,
            "probabilities": probabilities,
            "custom_time": custom_time
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"code": 500, "msg": str(e)}), 500


if __name__ == "__main__":
    # 启动服务，监听端口 20229
    app.run(host="0.0.0.0", port=20229)
