from pydantic import BaseModel
from typing import List

class VRData(BaseModel):
    start_stamp: str
    eye_id: str
    eyeposition: List[List[float]]
    eyerotation: List[List[float]]
    end_stamp: str
    score: int
    test_version: int
    end_timer: float
    initial_timer: float
    rotation_speed: float
    obj_rotation: float
