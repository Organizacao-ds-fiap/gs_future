# %% 
import tempfile
import pandas as pd

# %%
df = pd.read_csv("data/llm_matchmaker_dataset_1000.csv")
original_df = df.copy()
print(df.info())
df.sample(5)

# %%
target = "best_model"
features = [col for col in df.columns if col != target and not col.startswith("score_")]

print("Target:", target)
print("Features:", features)

# %%
X = df[features]
y = df[target]

# %%
categorical_features = [col for col in X.columns if col != "determinism_needed"]

categorical_features

# %%
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score, train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.base import clone
from tqdm import tqdm
# %%
pipeline = Pipeline(steps=[
    ("preprocessor", ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ]
    )),
    ("classifier", RandomForestClassifier(random_state=42, n_jobs=1)),
], memory=tempfile.mkdtemp())

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# %%
param_grid = {
    "classifier__n_estimators": [100, 300],
    "classifier__max_depth": [None, 10, 20],
    "classifier__min_samples_split": [2, 5, 10],
    "classifier__min_samples_leaf": [1, 2, 4],
    "classifier__max_features": ["sqrt", "log2", None],
    "classifier__bootstrap": [True, False],
    "classifier__max_leaf_nodes": [None, 50, 100],
}

# %%
N_SPLITS = 5
N_TRIALS = 15
N_ITER = 200
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)


# %%
all_results = []

random_search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_grid,
    n_iter=N_ITER, 
    cv=skf,
    n_jobs=-1,
    scoring="accuracy",
    random_state=42,
    verbose=2
)

random_search.fit(X_train, y_train)
print("Best Parameters:", random_search.best_params_)
print("Best CV Accuracy:", random_search.best_score_)

# %%
best_model = random_search.best_estimator_
best_model.fit(X_train, y_train)

# %%
y_true = y_test
y_pred = best_model.predict(X_test)
# %%
print(f"Test Set Classification Report: {classification_report(y_true, y_pred)}")
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_model.classes_)
disp.plot(cmap="Blues")

# %%
import joblib

# %%
best_model_filename = "best_llm_matchmaker_model.joblib"
joblib.dump(best_model, best_model_filename)
print(f"Best model saved to {best_model_filename}")