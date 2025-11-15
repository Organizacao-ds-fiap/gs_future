model_matcher = None

def load_models():
    import os
    import joblib
    
    global model_matcher

    main_path = os.path.dirname(__file__)
    MODEL_MATCHER_PATH = os.path.join(main_path, '..', 'models/best_llm_matchmaker_model.joblib')

    model_matcher = joblib.load(MODEL_MATCHER_PATH)