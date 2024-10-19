import asyncio
import datetime
import os
from sqlalchemy import *
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, Session
from contextlib import asynccontextmanager
from sqlalchemy_serializer import SerializerMixin
from sqlalchemy.orm import declarative_base
from backend.backend_server.log import log as logger

logger.info("工作空间{0}".format(os.getcwd()))
db_path = os.path.join(os.getcwd(), "task.sqlite")
logger.info("db path {0}".format(db_path))
async_engine = create_async_engine('sqlite+aiosqlite:///{0}'.format(db_path), echo=False,
                                   pool_pre_ping=True, connect_args={'check_same_thread': False}, pool_recycle=1800)
logger.info("current path {0}".format(os.getcwd()))
AsyncSessionLocal = sessionmaker(bind=async_engine, class_=AsyncSession, expire_on_commit=True,
                                 autocommit=False, autoflush=False)
Base = declarative_base()


@asynccontextmanager
async def async_connect():
    session = AsyncSessionLocal()
    try:
        logger.info("sql begin")
        yield session
        await session.commit()
        logger.info("sql success")
    except BaseException as e:
        await session.rollback()
        logger.error(e)
        raise e
    finally:
        logger.info("sql end")
        await session.close()


async def update_table_structure():
    async with async_engine.begin() as conn:
        # 反射现有的数据库结构
        await conn.run_sync(Base.metadata.create_all)


async def create_table():
    await update_table_structure()


class Task(Base, SerializerMixin):
    __tablename__ = 'tasks'
    id = Column(Integer, primary_key=True, autoincrement=True)
    start_time = Column(DateTime, default=None)
    end_time = Column(DateTime, default=None)
    status = Column(Integer)  # 0未开始, 1 执行中 , 2 执行完成
    file_dir = Column(String(255), default=None)  # 存储img文件的路径
    name = Column(String(255), default=None)  # 任务名称
    position = Column(String(255), default=None)  # 当前任务所处园区


class User(Base, SerializerMixin):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), default=None)  # 用户名称
    username = Column(String(255), default=None)  # 登录账号
    password = Column(String(255), default=None)  # 登录密码


class UserCRUD(object):
    @classmethod
    async def get_all_user(cls):
        async with async_connect() as session:
            session: Session
            async with session.begin():
                result = await session.execute(select(User))
                all_user = result.scalars().fetchall()
                result_list = [u.to_dict() for u in all_user]
                return result_list

    @classmethod
    async def get_item_user(cls, user_name):
        async with async_connect() as session:
            session: Session
            async with session.begin():
                result = await session.execute(
                    select(User).filter(
                        User.username == user_name,
                    )
                )
                item_user = result.scalars().first()
                return item_user.to_dict()

    @classmethod
    async def insert_item_user(cls, **params):
        async with async_connect() as session:
            session: Session
            async with session.begin():
                result = await session.add(User(**params))
                all_user = result.scalars().fetchall()
                result_list = [u.to_dict() for u in all_user]
                return result_list


class TaskCRUD(object):
    @classmethod
    async def set_task_running(cls, task_id, monitor_pid):
        async with async_connect() as session:
            session: Session
            async with session.begin():
                result = await session.execute(select(Task).filter(Task.id == task_id))
                task = result.scalars().first()
                assert task, "NOT FIND TASK"
                assert task.status == 0, "TASK RUNNING FAIL, TASK STATUS IS {0}".format(task.status)
                task.status = 1
                task.monitor_pid = monitor_pid
                return task.to_dict()

    @classmethod
    async def get_all_task(cls):
        async with async_connect() as session:
            session: Session
            async with session.begin():
                result = await session.execute(select(Task))
                task = result.scalars().fetchall()
                result_list = [t.to_dict() for t in task]
                result_list.sort(key=lambda x: x.get("start_time"), reverse=True)
                return result_list

    @classmethod
    async def get_item_task(cls, task_id):
        async with async_connect() as session:
            session: Session
            async with session.begin():
                result = await session.execute(select(Task).filter(Task.id == task_id))
                task = result.scalars().first()
                assert task, "NOT FIND TASK"
                return task.to_dict()

    @classmethod
    async def stop_task(cls, task_id):
        async with async_connect() as session:
            session: Session
            async with session.begin():
                result = await session.execute(select(Task).filter(Task.id == task_id))
                task = result.scalars().first()
                assert task, "NOT FIND TASK"
                assert task.status != 0, "TASK NOT RUNNING, TASK STATUS IS {0}".format(task.status)
                task.status = 2
                task.end_time = datetime.datetime.now()
                return task.to_dict()

    @classmethod
    async def delete_task(cls, task_id):
        async with async_connect() as session:
            session: Session
            async with session.begin():
                result = await session.execute(select(Task).filter(Task.id == task_id))
                task = result.scalars().first()
                assert task, "NOT FIND TASK"
                assert task.status != 1, "TASK RUNNING NOT DELETE, TASK STATUS IS {0}".format(task.status)
                res = task.to_dict()
                await session.delete(task)
                return res

    @classmethod
    async def change_task_name(cls, task_id, new_name):
        async with async_connect() as session:
            session: Session
            async with session.begin():
                result = await session.execute(select(Task).filter(Task.id == task_id))
                task = result.scalars().first()
                assert task, "NOT FIND TASK"
                task.name = new_name
                return task.to_dict()


asyncio.run(create_table())
