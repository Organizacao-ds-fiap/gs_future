# %%
import joblib
import pandas as pd

# %%
model_pipe = joblib.load("src/llm_matchmaker/apipredict/models/best_llm_matchmaker_model.joblib")
model_pipe

# %%
test_line = "classification,general,multi,cloud,cpu,high,0,high,creative".split(",")
test_line
model_params = ['task_type', 'domain', 'input_language','privacy_requirement','hardware_available','hallucination_tolerance', 'temperature_pref','output_style']

test_dict = dict(zip(model_params, test_line))
test_dict = pd.DataFrame([test_dict])
test_dict

# %%
model_pipe.steps
model_pipe.predict(test_dict)