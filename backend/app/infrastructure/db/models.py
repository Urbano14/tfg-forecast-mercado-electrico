from sqlalchemy import Column, DateTime, Float, Integer

from app.infrastructure.db.base import Base


class MarketData(Base):
    __tablename__ = "market_data"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)

    price = Column(Float, nullable=False)

    demand_forecast = Column(Float, nullable=True)
    wind_forecast = Column(Float, nullable=True)
    solar_forecast = Column(Float, nullable=True)
    hydro_programmed = Column(Float, nullable=True)