from pydantic import BaseModel
from typing import List
from datetime import datetime

# Define a Pydantic model for the API schema
class VRDataCreate(BaseModel):
    name: str
    description: str

class VRData(BaseModel):
    id: int
    session_id: str
    eyeposition: List[List[float]]
    eyerotation: List[List[float]]
    start_stamp: datetime
    end_stamp: datetime

    class Config:
        orm_mode = True  # Allows interaction between Pydantic and SQLAlchemy