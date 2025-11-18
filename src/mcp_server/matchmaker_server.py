from typing import Any
import requests
from mcp.server.fastmcp import FastMCP
from enums.params import TaskType, Domain, InputLanguage, PrivacyRequirement, HardwareAvailable, HallucinationTolerance, DeterminismNeeded, TemperaturePreference, OutputStyle
import logging

BASE_URL = "http://localhost:8080"

mcp = FastMCP("LLM_Matchmaker_Agent")

@mcp.resource("http://localhost/available_task_types")
def get_params_enum() -> dict[str, list[str]]:
    """
    Provides the enumerated options for each parameter to the AI.
    
    Returns:
        dict: A dictionary with parameter names as keys and lists of valid options as values.
    """
    params_enum = {
        "task_type": [e.value for e in TaskType],
        "domain": [e.value for e in Domain],
        "input_language": [e.value for e in InputLanguage],
        "privacy_requirement": [e.value for e in PrivacyRequirement],
        "hardware_available": [e.value for e in HardwareAvailable],
        "hallucination_tolerance": [e.value for e in HallucinationTolerance],
        "determinism_needed": [e.value for e in DeterminismNeeded],
        "temperature_preference": [e.value for e in TemperaturePreference],
        "output_style": [e.value for e in OutputStyle]
    }
    print(f"[MCP Server] Providing parameter enums to AI: {params_enum}")
    return params_enum

@mcp.tool()
def get_llm_recommendation(
    task_type: TaskType,
    domain: Domain,
    input_language: InputLanguage,
    privacy_requirement: PrivacyRequirement,
    hardware_available: HardwareAvailable,
    hallucination_tolerance: HallucinationTolerance,
    determinism_needed: DeterminismNeeded,
    temperature_preference: TemperaturePreference,
    output_style: OutputStyle
) -> dict | str:
    """
    Connects to the LLM Matchmaker API to get a model recommendation.
    
    This tool will call an external REST API to find the best LLM.
    
    Args:
        task_type (TaskType): The type of task. 
                        Options: "generation", "extraction", "reasoning", "classification", "summarization"
    domain (Domain): The subject matter. 
                    Options: "general", "legal", "technical", "finance", "medical", "ecommerce"
    input_language (InputLanguage): The language of the input. 
                            Options: "en", "pt", "multi"
    privacy_requirement (PrivacyRequirement): The privacy level needed. 
                                Options: "cloud", "local", "hybrid"
    hardware_available (HardwareAvailable): The available hardware. 
                                Options: "consumer_gpu", "cpu", "pro_gpu", "edge"
    hallucination_tolerance (HallucinationTolerance): The tolerance for inaccurate info. 
                                    Options: "high", "medium", "low"
    determinism_needed (DeterminismNeeded): Whether the output must be identical every time. 
                                Options: 1 (True) or 0 (False)
    temperature_preference (TemperaturePreference): The desired creativity. 
                                    Options: "low", "medium", "high"
    output_style (OutputStyle): The desired tone or format. 
                        Options: "formal", "creative", "factual", "precise"
    
    Returns:
        dict: A dictionary with the prediction, e.g., {"prediction": "Gemini"}
        str: An error message if the API call fails.
    """
    
    ENDPOINT = "/predict-match"
    
    # Pack the arguments into the dictionary for the API call
    query_params = {
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

    print(f"[MCP Server] Calling external API at {BASE_URL}{ENDPOINT}")
    print(f"[MCP Server] Params: {query_params}")
    
    try:
        # Make the GET request
        response = requests.get(f"{BASE_URL}{ENDPOINT}", params=query_params)
        response.raise_for_status()
        
        # --- Success ---
        result = response.json()
        print(f"[MCP Server] API Success. Returning to AI: {result}")
        # Return the successful JSON dictionary to the AI
        return result

    except requests.exceptions.HTTPError as http_err:
        error_msg = f"HTTP Error: {http_err}. Status: {response.status_code}. Details: {response.text}"
        print(f"[MCP Server] ERROR: {error_msg}")
        return error_msg # Return the error string to the AI
        
    except requests.exceptions.ConnectionError as conn_err:
        error_msg = f"Connection Error: Could not connect to {BASE_URL}. {conn_err}"
        print(f"[MCP Server] ERROR: {error_msg}")
        return error_msg
        
    except requests.exceptions.RequestException as err:
        error_msg = f"An Error Occurred: {err}"
        print(f"[MCP Server] ERROR: {error_msg}")
        return error_msg

# 3. Run the server
if __name__ == "__main__":
    print("Starting MCP server...")
    print(f"This server will act as a bridge to the API at: {BASE_URL}")
    print("An AI can connect and use the 'get_llm_recommendation' tool.")
    mcp.run()
