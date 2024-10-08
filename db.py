from sqlalchemy import create_engine, MetaData
from databases import Database

# SQLite connection (you can switch to PostgreSQL or MySQL here)
DATABASE_URL = "postgresql://localhost:5432/bigbrainsdb"

# Create an instance of the database
database = Database(DATABASE_URL)

# SQLAlchemy engine and metadata
engine = create_engine(DATABASE_URL)
metadata = MetaData()