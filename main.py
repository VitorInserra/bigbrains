from fastapi import FastAPI, HTTPException
from sqlalchemy import select
from db import database, engine, metadata
from models.models import vrdata
from entities.VRData import VRData, VRDataCreate

app = FastAPI()

metadata.create_all(engine)

@app.on_event("startup")
async def startup():
    await database.connect()

@app.on_event("shutdown")
async def shutdown():
    await database.disconnect()

@app.get("/ping")
async def ping():
    print("pong")
    return "pong"

@app.post("/vrdata/", response_model=VRData)
async def create_vrdata(data: VRDataCreate):
    query = vrdata.insert().values(name=data.name, description=data.description, value=data.value)
    last_record_id = await database.execute(query)
    return {**data.model_dump(), "id": last_record_id}

# Endpoint to get a VR data entry by ID
@app.get("/vrdata/{data_id}", response_model=VRData)
async def read_vrdata(data_id: int):
    query = select([vrdata]).where(vrdata.c.id == data_id)
    data = await database.fetch_one(query)
    if data is None:
        raise HTTPException(status_code=404, detail="VR Data not found")
    return data