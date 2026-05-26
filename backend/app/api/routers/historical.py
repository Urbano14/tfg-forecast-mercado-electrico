from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from datetime import datetime

from app.core.database import get_db
from app.schemas.historical import HistoricalDataResponse
from app.application.services.historical_service import (
    get_historical_data_between,
    get_historical_data_range,
)

# Rutas para obtener datos históricos.
router = APIRouter(prefix="/historical", tags=["historical"])

@router.get("", response_model=list[HistoricalDataResponse])
@router.get("/", response_model=list[HistoricalDataResponse])

# Obtener datos históricos entre dos fechas, con un límite opcional de resultados.
def get_historical_data(
    start: datetime,
    end: datetime,
    limit: int | None = Query(default=None, ge=1, le=10000),
    db: Session = Depends(get_db)
):
    if start >= end:
        raise HTTPException(
            status_code=400,
            detail="start must be earlier than end"
        )

    data = get_historical_data_between(
        db=db,
        start=start,
        end=end,
        limit=limit
    )

    return data

@router.get("/range")
def get_historical_range(db: Session = Depends(get_db)):
    return get_historical_data_range(db)
