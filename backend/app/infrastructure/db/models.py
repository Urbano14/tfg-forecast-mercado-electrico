from sqlalchemy import Column, DateTime, Float, Integer

from app.infrastructure.db.base import Base

#define el modelo ORM de SQLAlchemy para la tabla principal de la base de datos.
#Define cómo se representa en Python la tabla:market_data

class MarketData(Base):
    __tablename__ = "market_data"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)

    price = Column(Float, nullable=False)

    demand_forecast = Column(Float, nullable=True)
    wind_forecast = Column(Float, nullable=True)
    solar_forecast = Column(Float, nullable=True)
    hydro_programmed = Column(Float, nullable=True)