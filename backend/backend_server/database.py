import asyncio
import os

from sqlalchemy import *
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
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
    token = Column(String(255), default=None)  # 登录密码


asyncio.run(create_table())