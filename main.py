# import sqlite3
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
# from entities.VRData import VRData, VRDataCreate
from typing import List
import uvicorn
# from sqlalchemy import create_engine, MetaData
# from sqlalchemy.orm import sessionmaker


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# SQLite connection (you can switch to PostgreSQL or MySQL here)
# DATABASE_URL = "postgresql://localhost:5432/bigbrainsdb"

# SQLAlchemy engine and metadata
# engine = create_engine(DATABASE_URL)
# metadata = MetaData()

# def get_db():
#     db = SessionLocal()
#     try:
#         yield db
#     finally:
#         db.close()

@app.get("/ping")
async def ping():
    print("pong")
    return "pong"

# @app.post("/vrdata/")
# def create_vrdata(name: str, description: str, value: int):
#     conn = get_db()
#     cursor = conn.cursor()
#     cursor.execute(
#         "INSERT INTO vr_data (name, description, value) VALUES (?, ?, ?)",
#         (name, description, value),
#     )
#     conn.commit()
#     return {"id": cursor.lastrowid, "name": name, "description": description, "value": value}

# @app.get("/vrdata/{data_id}")
# def read_vrdata(data_id: int):
#     conn = get_db()
#     cursor = conn.cursor()
#     cursor.execute("SELECT * FROM vr_data WHERE id = ?", (data_id,))
#     result = cursor.fetchone()
#     if result is None:
#         raise HTTPException(status_code=404, detail="Data not found")
#     return dict(result)

@app.post("/datadump")
async def data_dump(position_data: List[List[float]]): #, rotation_data: List[List[float]]):
    print(position_data)
    # print(rotation_data)
    return "received"

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=80)