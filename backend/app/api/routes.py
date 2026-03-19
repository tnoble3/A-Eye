from fastapi import APIRouter, HTTPException
from app.api.schemas import AnalyzeRequest, AnalyzeResponse, Signal
from app.utils.image_io import fetch_image
from app.services.feature_layer import analyze_image
from app.services.aggregation import combine_scores

router = APIRouter()

@router.get("/health")
def health():
    return {"ok": True}

@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest):
    try:
        img = await fetch_image(str(req.image_url))
        feature_score = quickElaScore(img)
        cnn_score = None #placeholder for future CNN score
        final = combine_scores(feature_score, cnn_score)
        signals = [
            Signal(name="ela_placeholder", detail="Basic image variability proxy (replace with real ELA)", score=feature_score)
        ]
        return AnalyzeResponse(final_confidence=final, cnn_confidence=cnn_score, feature_confidence = feature_score ,signals=signals, meta={"version": "mvp-0"})
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))