from tinydb import TinyDB, Query
from tinydb.storages import MemoryStorage
from filelock import FileLock
from tinydb.table import Document

# 初始化 TinyDB
data_file = "data.json"
db_lock = FileLock("data.lock")  # 防止进程间竞争
db = TinyDB(data_file)


class IdentifySchema(dict):
    id: str
    record_time: int
    disease_type: str
    disease_type_rate: float
    disease_monitor_time: int
    img_str: str
    upload_user: str
    upload_solution: str


def get_identify_list(upload_user=None):
    with db_lock:  # 使用 with 语句来自动管理锁的获取和释放
        db_table = db.table("identify_record")
        items = list(db_table.all())  # 获取所有记录
        if upload_user:
            return [item for item in items if item.get('upload_user') == upload_user]
        return items


def add_item_identify(item: IdentifySchema):
    with db_lock:  # 使用 with 语句来自动管理锁的获取和释放
        db_table = db.table("identify_record")
        db_table.insert(item)  # 添加新记录


def del_items(ids: list[str]):
    with db_lock:  # 使用 with 语句来自动管理锁的获取和释放
        db_table = db.table("identify_record")
        query = Query()
        db_table.remove(
            doc_ids=[db_table.get(query.id == item).doc_id for item in ids if db_table.get(query.id == item)])
        return ids