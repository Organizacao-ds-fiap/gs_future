from fastapi import APIRouter, Depends
from utils.dependencies import get_logger
from services.predicts import predict_match
from results.match_result import ModelResult
from enums.params import TaskType, Domain, InputLanguage, PrivacyRequirement, HardwareAvailable, HallucinationTolerance, DeterminismNeeded, TemperaturePreference, OutputStyle

router = APIRouter(tags=["Predicts"])

@router.post("/predict-match", summary='Predicão com Melhor LLM', response_model=ModelResult)
async def Predict_Match(task_type: TaskType, domain: Domain, input_language: InputLanguage, privacy_requirement: PrivacyRequirement, hardware_available: HardwareAvailable, hallucination_tolerance: HallucinationTolerance, determinism_needed: DeterminismNeeded, temperature_preference: TemperaturePreference, output_style: OutputStyle, logger = Depends(get_logger)):
    return predict_match(task_type, domain, input_language, privacy_requirement, hardware_available, hallucination_tolerance, determinism_needed, temperature_preference, output_style, logger)