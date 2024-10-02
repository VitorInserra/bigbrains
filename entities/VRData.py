from pydantic import BaseModel

# Define a Pydantic model for the API schema
class VRDataCreate(BaseModel):
    name: str
    description: str

class VRData(BaseModel):
    id: int
    name: str
    description: str

    class Config:
        orm_mode = True  # Allows interaction between Pydantic and SQLAlchemy