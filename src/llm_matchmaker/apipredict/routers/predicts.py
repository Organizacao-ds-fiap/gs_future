from fastapi import APIRouter, Depends
from utils.dependencies import get_logger
# from results.model_result import ModelResult
from services.predicts import predict_match

router = APIRouter(tags=["Predicts"])

@router.post("/predict-match", summary='Predicão com Melhor LLM')
async def Predict_Match(logger = Depends(get_logger)):
    return predict_match(logger)