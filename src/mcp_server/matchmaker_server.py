from typing import Any
import httpx
from mcp.server.fastmcp import FastMCP
from enums.params import TaskType, Domain, InputLanguage, PrivacyRequirement, HardwareAvailable, HallucinationTolerance, DeterminismNeeded, TemperaturePreference, OutputStyle
from results.match_result import ModelResult, LLMs
import logging

# Initialize FastMCP server
mcp = FastMCP("MatchmakerServer",)

# Constants
BASE_URL = "http://localhost:8080"
USER_AGENT = "weather-app/1.0"

async def make_api_prediction(url: str, params: dict[str, Any], logger: logging.Logger) -> dict[str, Any] | None:
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/json"
        }
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, params=params, headers=headers)
            logger.debug(f"API Response Status: {response.status_code}")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error making API prediction: {e}")
            return None

@mcp.tool()
async def get_best_llm(
    task_type: TaskType,
    domain: Domain,
    input_language: InputLanguage,
    privacy_requirement: PrivacyRequirement,
    hardware_available: HardwareAvailable,
    hallucination_tolerance: HallucinationTolerance,
    determinism_needed: DeterminismNeeded,
    temperature_preference: TemperaturePreference,
    output_style: OutputStyle
) -> ModelResult | str:
    logger = logging.getLogger("MatchmakerServer")
    logger.info("Received request for best LLM match")

    params = {
        "task_type": task_type.value,
        "domain": domain.value,
        "input_language": input_language.value,
        "privacy_requirement": privacy_requirement.value,
        "hardware_available": hardware_available.value,
        "hallucination_tolerance": hallucination_tolerance.value,
        "determinism_needed": determinism_needed.value,
        "temperature_preference": temperature_preference.value,
        "output_style": output_style.value
    }

    logger.debug(f"Request parameters: {params}")

    api_url = f"{BASE_URL}/predict-match"
    api_response = await make_api_prediction(api_url, params, logger)

    if api_response and "prediction" in api_response:
        best_model_name = api_response["prediction"]
        logger.info(f"Best model determined: {best_model_name}")
        best_model = LLMs[best_model_name]
        return ModelResult(model=best_model,)
    else:
        logger.error("Failed to retrieve best model from API response")
        return "Failed to determine the best LLM model. Please try again later."

def main():
    # Initialize and run the server
    # mcp.run(transport='stdio')
    mcp.run()

if __name__ == "__main__":
    main()
