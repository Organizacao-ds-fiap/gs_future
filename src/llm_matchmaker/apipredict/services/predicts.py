from fastapi import HTTPException

def predict_match(logger):
    try:
        logger.info("Predict Match called")
        # Dummy implementation for illustration
        best_llm = {
            "model": "Llama-3-70B",
            "reason": "Best suited for extraction and classification tasks."
        }
        logger.info(f"Best LLM selected: {best_llm['model']}")
        return best_llm
    except Exception as e:
        logger.error(f"Error in Predict Match: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")