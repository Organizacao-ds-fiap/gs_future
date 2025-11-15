from fastapi import HTTPException
import pandas as pd

from enums.params import TaskType, Domain, InputLanguage, PrivacyRequirement, HardwareAvailable, HallucinationTolerance, DeterminismNeeded, TemperaturePreference, OutputStyle
from results.match_result import ModelResult, LLMs
import utils.models_loader as models

def predict_match(task_type: TaskType, domain: Domain, input_language: InputLanguage, privacy_requirement: PrivacyRequirement, hardware_available: HardwareAvailable, hallucination_tolerance: HallucinationTolerance, determinism_needed: DeterminismNeeded, temperature_preference: TemperaturePreference, output_style: OutputStyle, logger):
    try:
        MODEL_PARAMS = ['task_type',
                        'domain',
                        'input_language',
                        'privacy_requirement',
                        'hardware_available',
                        'hallucination_tolerance',
                        'temperature_pref',
                        'output_style'
                        ]
        pred_matrix = pd.DataFrame([{
            'task_type': task_type.value,
            'domain': domain.value,
            'input_language': input_language.value,
            'privacy_requirement': privacy_requirement.value,
            'hardware_available': hardware_available.value,
            'hallucination_tolerance': hallucination_tolerance.value,
            'temperature_pref': temperature_preference.value,
            'output_style': output_style.value
        }], columns=MODEL_PARAMS)
        
        logger.debug(f"Prediction Matrix: {pred_matrix}")
        
        prediction = models.model_matcher.predict(pred_matrix)
        
        logger.info("Predict Match called")
        
        best_llm = ModelResult(
            prediction=LLMs(prediction[0])
        )
        return best_llm
    except Exception as e:
        logger.error(f"Error in Predict Match: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")