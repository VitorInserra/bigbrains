from sqlalchemy import Column, Integer, String, ARRAY, TIMESTAMP, Float, func
from sqlalchemy.ext.declarative import declarative_base
from db import engine

Base = declarative_base()

class MuseDataModel(Base):
    __tablename__ = "muse_data"

    session_id = Column(String, primary_key=True, index=True)
    start_stamp = Column(TIMESTAMP(timezone=True))
    tp9 = Column(ARRAY(Float))
    af7 = Column(ARRAY(Float))
    af8 = Column(ARRAY(Float))
    tp10 = Column(ARRAY(Float))
    end_stamp = Column(TIMESTAMP(timezone=True))

Base.metadata.create_all(bind=engine)
