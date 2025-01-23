from sqlalchemy import Column, Integer, String, ARRAY, TIMESTAMP, Float, func
from sqlalchemy.ext.declarative import declarative_base
from db import engine

Base = declarative_base()


class VRDataModel(Base):
    __tablename__ = "vr_data"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String)
    eyeposition = Column(ARRAY(Float))
    eyerotation = Column(ARRAY(Float))
    eye_id = Column(String)
    start_stamp = Column(TIMESTAMP(timezone=True), server_default=func.now())
    end_stamp = Column(TIMESTAMP(timezone=True))
    current_score = Column(Integer)
    rotation_quantity = Column(Integer)

Base.metadata.create_all(bind=engine)