from datetime import datetime, timedelta

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.infrastructure.db.models import MarketData


def get_historical_data_between(
    db: Session,
    start: datetime,
    end: datetime,
    limit: int | None = None
):
    query = (
        db.query(MarketData)
        .filter(MarketData.timestamp >= start)
        .filter(MarketData.timestamp <= end)
        .order_by(MarketData.timestamp.asc())
    )

    if limit is not None:
        query = query.limit(limit)

    return query.all()


def get_historical_data_range(db: Session):
    min_timestamp, max_timestamp = (
        db.query(
            func.min(MarketData.timestamp),
            func.max(MarketData.timestamp)
        )
        .one()
    )

    return {
        "start": min_timestamp,
        "end": max_timestamp
    }


def get_previous_24_hours(db: Session, requested_date: datetime):
    start = requested_date - timedelta(hours=24)

    return (
        db.query(MarketData)
        .filter(MarketData.timestamp >= start)
        .filter(MarketData.timestamp < requested_date)
        .order_by(MarketData.timestamp.asc())
        .all()
    )