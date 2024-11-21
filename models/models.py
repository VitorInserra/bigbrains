# models.py
from sqlalchemy import Column, Integer, String, ARRAY, TIMESTAMP, Float, func
from sqlalchemy.ext.declarative import declarative_base
from db import engine  # Import engine from db.py

# Define the base class for models
Base = declarative_base()

# Declare the VRDataModel class
class VRDataModel(Base):
    __tablename__ = "vr_data"  # Name of the table in the database

    # Define columns
    id = Column(Integer, primary_key=True, index=True)  # Primary key
    eyeposition = Column(ARRAY(Float))  # Store eye position data (list of floats)
    eyerotation = Column(ARRAY(Float))  # Store eye rotation data (list of floats)
    start = Column(TIMESTAMP(timezone=True), server_default=func.now())  # Auto-generate timestamp
    end = Column(TIMESTAMP(timezone=True))

# Create the table if it doesn't already exist
Base.metadata.create_all(bind=engine)