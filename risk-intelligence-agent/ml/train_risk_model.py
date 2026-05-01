"""Train a baseline model to predict risk levels."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def _resolve_csv_path(base_dir: Path) -> Path:
    """Find risk score CSV in Spark output directory or fallback plain file."""
    scores_dir = base_dir / 'data' / 'processed' / 'risk_scores.csv'
    if scores_dir.is_file():
        return scores_dir
    csv_files = sorted(scores_dir.glob('part-*.csv'))
    if not csv_files:
        raise FileNotFoundError(f'No CSV part file found in {scores_dir}')
    return csv_files[0]


def main():
    base_dir = Path(__file__).resolve().parents[1]
    model_dir = base_dir / 'models'
    model_dir.mkdir(parents=True, exist_ok=True)

    scores_csv = _resolve_csv_path(base_dir)
    df = pd.read_csv(scores_csv)

    feature_cols = ['route', 'origin', 'destination', 'delay_rate', 'weather_risk', 'supplier_reliability']
    target_col = 'risk_level'

    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    categorical_cols = ['route', 'origin', 'destination']
    numerical_cols = ['delay_rate', 'weather_risk', 'supplier_reliability']

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
            ('num', 'passthrough', numerical_cols),
        ]
    )

    model = Pipeline(
        steps=[
            ('preprocess', preprocessor),
            ('classifier', RandomForestClassifier(n_estimators=200, random_state=42)),
        ]
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print('Model Accuracy:', round(accuracy_score(y_test, preds), 4))
    print('Classification Report:')
    print(classification_report(y_test, preds))

    model_path = model_dir / 'risk_model.pkl'
    joblib.dump(model, model_path)
    print(f'Model saved to {model_path}')


if __name__ == '__main__':
    main()
