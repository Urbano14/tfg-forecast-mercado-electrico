from pydantic import BaseModel


class ModelInfoResponse(BaseModel):
    id: str
    name: str
    type: str
    horizon_hours: int


class ModelsListResponse(BaseModel):
    models: list[ModelInfoResponse]