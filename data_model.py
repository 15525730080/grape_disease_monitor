
from tinydb import TinyDB, Query
from tinydb.storages import MemoryStorage
from filelock import FileLock

# 初始化 TinyDB
db_lock = FileLock("data.lock")  # 防止进程间竞争
db = TinyDB(storage=MemoryStorage)


class IdentifySchema(dict):
    key_user_time: str
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
