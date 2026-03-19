from pydantic import BaseModel, Field, HttpUrl
from typing import Optional, List,Dict, Any

class AnalyzeRequest(BaseModel):
    imapge_url: HttpUrl
    
class Signal(BaseModel):
    name : str
    detail : str
    score : float = Field(ge=0.0, le=1.0)
    
class AnalyzeResponse(BaseModel):
    final_confidence: float = Field(ge=0.0, le=1.0)
    cnn_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    signals: List[Signal]
    meta: Dict[str, Any] = {}