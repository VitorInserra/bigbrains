from sqlalchemy import Column, Integer, String, ARRAY, TIMESTAMP, Float, func
from sqlalchemy.ext.declarative import declarative_base
from db import engine

Base = declarative_base()


class VRDataModel(Base):
    __tablename__ = "vr_data"
    id = Column(Integer, primary_key=True, autoincrement=True)

    session_id = Column(String)
    eye_interactables = Column(ARRAY(Float))
    eye_id = Column(String)
    start_stamp = Column(TIMESTAMP(timezone=True), server_default=func.now())
    end_stamp = Column(TIMESTAMP(timezone=True))
    score = Column(Integer)
    test_version = Column(Integer)
    end_timer = Column(Float)
    initial_timer = Column(Float)
    rotation_speed = Column(Float)
    obj_rotation = Column(Float)
    description = Column(String)


Base.metadata.create_all(bind=engine)
