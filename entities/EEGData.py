from datetime import datetime
import pandas as pd
from sqlalchemy.orm import Session
from models.models import MuseDataModel


def insert_eeg_db(db: Session):
    scnr = pd.read_csv("session_data.csv")
    df = pd.DataFrame(scnr)
    try:
        session_dump = {'start_stamp': datetime.fromtimestamp(df.iloc[0]['timestamp']), 'end_stamp': datetime.fromtimestamp(df.iloc[-1]['timestamp'])}
        for key in df.columns:
            if key != "timestamp":
                session_dump[key] = list(df.xs(key, axis=1))


        eeg_data = MuseDataModel(
            start_stamp=session_dump['start_stamp'],
            tp9=session_dump['tp9'],
            af7=session_dump['af7'],
            af8=session_dump['af8'],
            tp10=session_dump['tp10'],
            end_stamp=session_dump['end_stamp'],
        )
        

        db.add(eeg_data)
        db.commit()
        db.refresh(eeg_data)
    except Exception as e:
        print(f"No data: {str(e)}")

    f = open("session_data.csv", "w")
    f.truncate()
    f.write("timestamp,tp9,af7,af8,tp10,hr\n")
    f.close()
    print("Saved to eeg to database")
