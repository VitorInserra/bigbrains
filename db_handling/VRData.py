from pydantic import BaseModel
from typing import List
from datetime import datetime

# Define a Pydantic model for the API schema
class VRDataCreate(BaseModel):
    name: str
    description: str

class VRData(BaseModel):
    id: int
    name: str
    eyeposition: List[List[float]]
    eyerotation: List[List[float]]
    timestamp: datetime
    current_score: int
    rotation_quantity: List[List[float]]

    class Config:
        orm_mode = True  # Allows interaction between Pydantic and SQLAlchemy

class DataDumpRequest(BaseModel):
    start_stamp: datetime
    eye_id: str
    position_data: List[List[float]]
    rotation_data: List[List[float]]
    end_stamp: datetime
    current_score: int
    rotation_quantity: List[List[float]]
    