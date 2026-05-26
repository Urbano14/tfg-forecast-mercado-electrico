from app.core.config import settings
from app.core.database import engine
from app.infrastructure.db.base import Base
from app.infrastructure.db import init_models

# Crea las tablas en la base de datos.

def create_tables():
    print("DATABASE_URL cargada:", repr(settings.DATABASE_URL))
    Base.metadata.create_all(bind=engine)


if __name__ == "__main__":
    create_tables()