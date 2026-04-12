from typing import Any
from pydantic import BaseModel, Field, HttpUrl


class AnalyzeRequest(BaseModel):
    #keeping a 'snake_case' in the API avoids alias churn while the service contract is stabilizing
    image_url: HttpUrl
class Signal(BaseModel):
    name: str
    detail: str
    score: float = Field(ge=0.0, le=1.0)
class AnalyzeResponse(BaseModel):
    final_confidence: float = Field(ge=0.0, le=1.0)
    cnn_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    feature_confidence: float = Field(ge=0.0, le=1.0)
    signals: list[Signal]
    meta: dict[str, Any] = Field(default_factory=dict)
