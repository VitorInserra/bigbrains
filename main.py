# main.py
from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
from db import SessionLocal
from models.models import VRDataModel  # Import your model from models.py
from pydantic import BaseModel
from typing import List
from datetime import datetime
import uvicorn
from muselsl import stream, list_muses, record
import multiprocessing
import pandas as pd
import os

app = FastAPI()


# Dependency to get the database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Define the Pydantic model for the incoming request
class DataDumpRequest(BaseModel):
    position_data: List[List[float]]
    rotation_data: List[List[float]]


@app.post("/datadump")
async def data_dump(data: DataDumpRequest, db: Session = Depends(get_db)):
    # Flatten the position_data and rotation_data lists
    print(data.position_data)
    flattened_eyeposition = [item for item in data.position_data]
    flattened_eyerotation = [item for item in data.rotation_data]

    # Create an instance of VRDataModel
    vr_data = VRDataModel(
        eyeposition=flattened_eyeposition,
        eyerotation=flattened_eyerotation,
        start=datetime.now(),
    )

    # Add the new record to the database
    db.add(vr_data)
    db.commit()
    db.refresh(vr_data)  # Refresh to get the generated ID

    return vr_data


@app.get("/getdata")
async def get_data(db: Session = Depends(get_db)):
    results = db.query(VRDataModel).all()
    for record in results:
        print(f"ID: {record.id}")
        print(f"Eye Position: {record.eyeposition}")
        print(f"Eye Rotation: {record.eyerotation}")


def stream_muse():
    muses = list_muses()
    stream(muses[0]["address"])


@app.get("/stream")
async def print_stream():
    directory = os.getcwd()
    filename = os.path.join(directory, "session_data.csv")
    record(duration=10, filename=filename)

    # md = MetaData("Symbol", "Name", "IPO Year", "Sector", "Industry")

    # scnr = pd.read_csv()
    # df = pd.DataFrame(scnr)

    # for i in range(len(df)):
    #     print(i)
    #     temp = df.loc[i]

    #     symbol = temp[md.symbol]
    #     symbol = edit_names(symbol)

    #     name = temp[md.name]
    #     name = edit_names(name)

    #     ipo_year = temp[md.ipo_year]
    #     ipo_year = edit_names(ipo_year)

    #     sector = temp[md.sector]
    #     sector = edit_names(sector)

    #     industry = temp[md.industry]
    #     industry = edit_names(industry)

    #     cur.execute(f"select from stock where symbol = '{symbol}'")
    #     if cur.fetchone() == None:
    #         cur.execute(
    #             f"insert into stock (id, frequency, symbol, name, ipoyear, sector, industry) values ('{i}', '{0}', '{symbol}', '{name}', '{ipo_year}', '{sector}', '{industry}')"
    #         )

    # conn.commit()

    # cur.close()
    # conn.close()


if __name__ == "__main__":
    p = multiprocessing.Process(target=stream_muse)
    p.start()
    uvicorn.run("main:app", host="0.0.0.0", port=80)
