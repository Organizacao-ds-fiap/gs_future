from fastapi import HTTPException
from enums.params import TaskType, Domain, InputLanguage, PrivacyRequirement, HardwareAvailable, HallucinationTolerance, DeterminismNeeded, TemperaturePreference, OutputStyle
from results.match_result import ModelResult, LLMs


def predict_match(task_type: TaskType, domain: Domain, input_language: InputLanguage, privacy_requirement: PrivacyRequirement, hardware_available: HardwareAvailable, hallucination_tolerance: HallucinationTolerance, determinism_needed: DeterminismNeeded, temperature_preference: TemperaturePreference, output_style: OutputStyle, logger):
    try:
        logger.info("Predict Match called")
        # Dummy implementation for illustration
        best_llm = ModelResult(
            probability=0.95,
            prediction=LLMs.GPT_4O
        )
        logger.info(f"Best LLM selected: {best_llm.prediction} with probability {best_llm.probability}")
        return best_llm
    except Exception as e:
        logger.error(f"Error in Predict Match: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")