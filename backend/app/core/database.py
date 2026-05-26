from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import settings

engine = create_engine(settings.DATABASE_URL) #objeto principal de SQLAlchemy para conectarse a PostgreSQL.
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine) #Permite crear sesiones cuando haga falta.

# obtener una sesión de base de datos. 
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()