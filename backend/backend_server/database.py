import asyncio
import os
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


asyncio.run(create_table())