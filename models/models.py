# models.py
from sqlalchemy import Column, Integer, String, ARRAY, TIMESTAMP, Float, func
from sqlalchemy.ext.declarative import declarative_base
from db import engine  # Import engine from db.py

# Define the base class for models
Base = declarative_base()

# Declare the VRDataModel class
class VRDataModel(Base):
    __tablename__ = "vr_data"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String)
    eyeposition = Column(ARRAY(Float))
    eyerotation = Column(ARRAY(Float))
    eye_id = Column(String)
    start_stamp = Column(TIMESTAMP(timezone=True), server_default=func.now())
    end_stamp = Column(TIMESTAMP(timezone=True))

class MuseDataModel(Base):
    __tablename__ = "muse_data"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String)
    start_stamp = Column(TIMESTAMP(timezone=True))
    tp9 = Column(ARRAY(Float))
    af7 = Column(ARRAY(Float))
    af8 = Column(ARRAY(Float))
    tp10 = Column(ARRAY(Float))
    end_stamp = Column(TIMESTAMP(timezone=True))


Base.metadata.create_all(bind=engine)