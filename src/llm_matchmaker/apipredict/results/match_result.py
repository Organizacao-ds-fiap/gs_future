from pydantic import BaseModel
from enum import Enum

class LLMs(Enum):
    GEMINI = "Gemini"
    DEEPSEEK = "Deepseek"
    LLAMA_3_70B = "Llama-3-70B"
    CLAUDE_2 = "Claude-2"
    GPT_4O = "GPT-4o"

class ModelResult(BaseModel):
    probability: float
    prediction: LLMs

