from sqlalchemy import Column, DateTime, Integer, Float, String, ARRAY
from sqlalchemy.ext.declarative import declarative_base
from db import engine

Base = declarative_base()

class EpocXDataModel(Base):
    __tablename__ = "epoc_x_data"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, index=True)
    start_stamp = Column(DateTime)
    end_stamp = Column(DateTime)
    # One approach is to store each channel’s data as an ARRAY of floats
    af3 = Column(ARRAY(Float))
    f7 = Column(ARRAY(Float))
    f3 = Column(ARRAY(Float))
    fc5 = Column(ARRAY(Float))
    t7 = Column(ARRAY(Float))
    p7 = Column(ARRAY(Float))
    o1 = Column(ARRAY(Float))
    o2 = Column(ARRAY(Float))
    p8 = Column(ARRAY(Float))
    t8 = Column(ARRAY(Float))
    fc6 = Column(ARRAY(Float))
    f4 = Column(ARRAY(Float))
    f8 = Column(ARRAY(Float))
    af4 = Column(ARRAY(Float))

    from sqlalchemy import Column, DateTime, Integer, Float, String, ARRAY
    from sqlalchemy.ext.declarative import declarative_base
    Base = declarative_base()
    
    class EpocXDataModel(Base):
        __tablename__ = "epoc_x_data"
    
        id = Column(Integer, primary_key=True, index=True)
        session_id = Column(String, index=True)
        start_stamp = Column(DateTime)
        end_stamp = Column(DateTime)
        # One approach is to store each channel’s data as an ARRAY of floats
        af3 = Column(ARRAY(Float))
        f7 = Column(ARRAY(Float))
        f3 = Column(ARRAY(Float))
        fc5 = Column(ARRAY(Float))
        t7 = Column(ARRAY(Float))
        p7 = Column(ARRAY(Float))
        o1 = Column(ARRAY(Float))
        o2 = Column(ARRAY(Float))
        p8 = Column(ARRAY(Float))
        t8 = Column(ARRAY(Float))
        fc6 = Column(ARRAY(Float))
        f4 = Column(ARRAY(Float))
        f8 = Column(ARRAY(Float))
        af4 = Column(ARRAY(Float))
    
Base.metadata.create_all(bind=engine)