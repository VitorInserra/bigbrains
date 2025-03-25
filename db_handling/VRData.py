from pydantic import BaseModel
from typing import List


class VRData(BaseModel):
    start_stamp: str
    eye_id: str
    eye_interactables: List[
        List[int]
    ]  # [0 -> Timer, 1 -> GameObject, 2 -> OutlineObject, 3 -> Score]:
    end_stamp: str
    score: int
    test_version: int
    end_timer: float
    initial_timer: float
    rotation_speed: float
    obj_rotation: float
    expected_rotation: float
    obj_size: float
