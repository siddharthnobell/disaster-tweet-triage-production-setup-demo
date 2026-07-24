from typing import Optional

from pydantic import BaseModel, Field


class TweetRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Raw tweet text")
    keyword: Optional[str] = Field(None, description="Disaster-related keyword, if any")


class PredictResponse(BaseModel):
    label: int = Field(..., description="1 = real disaster, 0 = not a real disaster")
    probability: float = Field(..., description="Model probability that this is a real disaster")
    model_name: str


class HealthResponse(BaseModel):
    status: str
    model_name: str
