from fastapi import APIRouter, HTTPException

from app.api.schemas import AnalyzeRequest, AnalyzeResponse, Signal
from app.services.pipeline import run_detection_pipeline
from app.utils.image_io import fetch_image

router = APIRouter()


@router.get("/health")
def health():
    return {
        "ok": True,
        "service": "a-eye-api",
        "architecture": "hybrid_cnn_feature",
        "client": "browser_extension",
    }


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest):
    try:
        image = await fetch_image(str(request.image_url))
        result = run_detection_pipeline(image)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    #this route stays intentionally thin so model experiments can change in the
    #a simple service layer without repeatedly reshaping the HTTP contract
    return AnalyzeResponse(
        final_confidence=result.final_confidence,
        cnn_confidence=result.cnn_confidence,
        feature_confidence=result.feature_confidence,
        signals=[Signal(**signal) for signal in result.signals],
        meta=result.meta,
    )
