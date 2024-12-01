import base64
import subprocess
import sys
import threading
import time
import traceback
from io import BytesIO
from simpledb import Client

from flask import Flask, request, jsonify
from flask_cors import CORS
from predict.img_diff import ImageComparator
from predict.launch_ensemble_classifier import ensemble_predict

# 初始化 Flask 应用
app = Flask(__name__)
CORS(app)  # 启用跨域
client = Client(port=20221)

def diff_before_img(img):
    res = client.get("before_img")
    if res:
        if time.time() - res.get("time") > 60 * 30:
            client.set("before_img", {"img": img, "time": time.time()})
        else:
            try:
                is_similar, similarity_score = ImageComparator.compare_images(res.get("img"), img,
                                                                              similarity_threshold=0.8)
                if is_similar:
                    print(f"The images are similar with a similarity score of {similarity_score:.2f}.")
                    return True
                else:
                    print(f"The images are not similar. Similarity score: {similarity_score:.2f}.")
                    client.set("before_img", {"img": img, "time": time.time()})
                    return False
            except:
                client.set("before_img", {"img": img, "time": time.time()})
                return False
    else:
        client.set("before_img", {"img": img, "time": time.time()})
        return False


@app.route("/keep-live/", methods=["GET"])
def keep_live():
    return 200


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
    def start_share_memory():
        p = subprocess.run([sys.executable, "-m", "simpledb", "-p", "20221"])
    threading.Thread(target=start_share_memory).start()
    # 启动服务，监听端口 20229
    app.run(host="0.0.0.0", port=20229)
