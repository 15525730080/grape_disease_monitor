import base64
import time
import traceback
import uuid
from io import BytesIO

from flask import Flask, request, jsonify
from flask_cors import CORS
from predict.img_diff import ImageComparator
from predict.launch_ensemble_classifier import ensemble_predict
from data_model import db, Query, IdentifySchema, db_lock, get_identify_list, add_item_identify

# 初始化 Flask 应用
app = Flask(__name__)
CORS(app)  # 启用跨域


def diff_before_img(img, img_encoded):
    """
    比较当前图片和数据库中存储的图片，并决定是否更新图片。
    """
    with db_lock:  # 确保进程间安全
        db_table = db.table("before_img")
        record = db_table.get(Query().key == "before_img")
        if record:
            if time.time() - record["time"] > 60 * 30:
                db_table.upsert({"key": "before_img", "img": img_encoded, "time": time.time()},
                                Query().key == "before_img")
            else:
                try:
                    is_similar, similarity_score = ImageComparator.compare_images(bytes(record["img"]), img,
                                                                                  similarity_threshold=0.8)
                    if is_similar:
                        print(f"The images are similar with a similarity score of {similarity_score:.2f}.")
                        return True
                    else:
                        print(f"The images are not similar. Similarity score: {similarity_score:.2f}.")
                        db_table.upsert({"key": "before_img", "img": img_encoded, "time": time.time()},
                                        Query().key == "before_img")

                        return False
                except Exception as e:
                    print(f"Error comparing images: {e}")
                    db_table.upsert({"key": "before_img", "img": img_encoded, "time": time.time()},
                                    Query().key == "before_img")
                    return False
        else:
            db_table.insert({"key": "before_img", "img": img_encoded, "time": time.time()})
            return False


@app.route("/keep-live/", methods=["GET"])
def keep_live():
    return "OK", 200


@app.route('/upload_base64/', methods=['POST'])
def upload_base64():
    try:
        data = request.get_json()
        user = data.get("user")
        base64_image = data.get("image", "")
        upload_solution = data.get("upload_solution", "")

        # 提取 Base64 部分
        header, encoded = base64_image.split(",", 1)
        image_data = base64.b64decode(encoded)
        image_data = bytes(image_data)
        is_similar = diff_before_img(image_data, encoded)
        predicted_class, probabilities, custom_time = ensemble_predict(BytesIO(image_data))
        if not is_similar and probabilities > 0.9:
            add_item_identify(IdentifySchema(
                id=str(uuid.uuid1()),
                disease_type=predicted_class,
                disease_type_rate=probabilities,
                disease_monitor_time=custom_time,
                record_time=int(time.time()),
                img_str=encoded,
                upload_user=user,
                upload_solution=upload_solution))
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


@app.route('/get_grape_disease_list/', methods=['GET'])
def get_grape_disease_list():
    return jsonify(get_identify_list())


if __name__ == "__main__":
    # 启动服务，监听端口 20229
    app.run(host="0.0.0.0", port=20229)
