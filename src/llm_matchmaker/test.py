# %%
import pandas as pd

# %%
df = pd.read_csv("data/llm_matchmaker_dataset_1000.csv")

cols_analyse = [col for col in df.columns if not col.startswith("score_")]
print(cols_analyse)
for col in cols_analyse:
    print(f"{col}: {df[col].nunique()} unique values \n {df[col].value_counts()}\n")