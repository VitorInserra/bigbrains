from sqlalchemy import Column, DateTime, Integer, Float, String, ARRAY
from sqlalchemy.ext.declarative import declarative_base
from db import engine

Base = declarative_base()

class EpocXDataModel(Base):
    __tablename__ = "epoc_x_data"

    id = Column(Integer, primary_key=True, autoincrement=True)

    session_id = Column(String)
    start_stamp = Column(DateTime)
    end_stamp = Column(DateTime)

    # AF3
    af3_theta = Column(ARRAY(Float))
    af3_alpha = Column(ARRAY(Float))
    af3_beta_l = Column(ARRAY(Float))
    af3_beta_h = Column(ARRAY(Float))
    af3_gamma = Column(ARRAY(Float))

    # F7
    f7_theta = Column(ARRAY(Float))
    f7_alpha = Column(ARRAY(Float))
    f7_beta_l = Column(ARRAY(Float))
    f7_beta_h = Column(ARRAY(Float))
    f7_gamma = Column(ARRAY(Float))

    # F3
    f3_theta = Column(ARRAY(Float))
    f3_alpha = Column(ARRAY(Float))
    f3_beta_l = Column(ARRAY(Float))
    f3_beta_h = Column(ARRAY(Float))
    f3_gamma = Column(ARRAY(Float))

    # FC5
    fc5_theta = Column(ARRAY(Float))
    fc5_alpha = Column(ARRAY(Float))
    fc5_beta_l = Column(ARRAY(Float))
    fc5_beta_h = Column(ARRAY(Float))
    fc5_gamma = Column(ARRAY(Float))

    # T7
    t7_theta = Column(ARRAY(Float))
    t7_alpha = Column(ARRAY(Float))
    t7_beta_l = Column(ARRAY(Float))
    t7_beta_h = Column(ARRAY(Float))
    t7_gamma = Column(ARRAY(Float))

    # P7
    p7_theta = Column(ARRAY(Float))
    p7_alpha = Column(ARRAY(Float))
    p7_beta_l = Column(ARRAY(Float))
    p7_beta_h = Column(ARRAY(Float))
    p7_gamma = Column(ARRAY(Float))

    # O1
    o1_theta = Column(ARRAY(Float))
    o1_alpha = Column(ARRAY(Float))
    o1_beta_l = Column(ARRAY(Float))
    o1_beta_h = Column(ARRAY(Float))
    o1_gamma = Column(ARRAY(Float))

    # O2
    o2_theta = Column(ARRAY(Float))
    o2_alpha = Column(ARRAY(Float))
    o2_beta_l = Column(ARRAY(Float))
    o2_beta_h = Column(ARRAY(Float))
    o2_gamma = Column(ARRAY(Float))

    # P8
    p8_theta = Column(ARRAY(Float))
    p8_alpha = Column(ARRAY(Float))
    p8_beta_l = Column(ARRAY(Float))
    p8_beta_h = Column(ARRAY(Float))
    p8_gamma = Column(ARRAY(Float))

    # T8
    t8_theta = Column(ARRAY(Float))
    t8_alpha = Column(ARRAY(Float))
    t8_beta_l = Column(ARRAY(Float))
    t8_beta_h = Column(ARRAY(Float))
    t8_gamma = Column(ARRAY(Float))

    # FC6
    fc6_theta = Column(ARRAY(Float))
    fc6_alpha = Column(ARRAY(Float))
    fc6_beta_l = Column(ARRAY(Float))
    fc6_beta_h = Column(ARRAY(Float))
    fc6_gamma = Column(ARRAY(Float))

    # F4
    f4_theta = Column(ARRAY(Float))
    f4_alpha = Column(ARRAY(Float))
    f4_beta_l = Column(ARRAY(Float))
    f4_beta_h = Column(ARRAY(Float))
    f4_gamma = Column(ARRAY(Float))

    # F8
    f8_theta = Column(ARRAY(Float))
    f8_alpha = Column(ARRAY(Float))
    f8_beta_l = Column(ARRAY(Float))
    f8_beta_h = Column(ARRAY(Float))
    f8_gamma = Column(ARRAY(Float))

    # AF4
    af4_theta = Column(ARRAY(Float))
    af4_alpha = Column(ARRAY(Float))
    af4_beta_l = Column(ARRAY(Float))
    af4_beta_h = Column(ARRAY(Float))
    af4_gamma = Column(ARRAY(Float))
    
Base.metadata.create_all(bind=engine)