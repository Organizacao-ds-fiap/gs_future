from enum import Enum

class TaskType(str, Enum):
    GENERATION = "generation"
    EXTRACTION = "extraction"
    REASONING = "reasoning"
    CLASSIFICATION = "classification"
    SUMMARIZATION = "summarization"

class Domain(str, Enum):
    GENERAL = "general"
    LEGAL = "legal"
    TECHNICAL = "technical"
    FINANCE = "finance"
    MEDICAL = "medical"
    ECOMMERCE = "ecommerce"
    
class InputLanguage(str, Enum):
    ENGLISH = "en"
    PORTUGUESE = "pt"
    MULTI = "multi"

class PrivacyRequirement(str, Enum):
    CLOUD = "cloud"
    LOCAL = "local"
    HYBRID = "hybrid"

class HardwareAvailable(str, Enum):
    CONSUMER_GPU = "consumer_gpu"
    CPU = "cpu"
    PRO_GPU = "pro_gpu"
    EDGE = "edge"

class HallucinationTolerance(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class DeterminismNeeded(str, Enum):
    HIGH = "high"
    LOW = "low"

class TemperaturePreference(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class OutputStyle(str, Enum):
    FORMAL = "formal"
    CREATIVE = "creative"
    FACTUAL = "factual"
    PRECISE = "precise"