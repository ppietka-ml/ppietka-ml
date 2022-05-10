"""
Predictive maintenance model for industrial equipment.
Detects early signs of failure using sensor time-series data and ML classifiers.
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score


@dataclass
class MaintenanceAlert:
    equipment_id: str
    failure_probability: float
    risk_level: str
    recommended_action: str
    features_triggered: list


def extract_features(sensor_df: pd.DataFrame) -> pd.DataFrame:
    """Extract statistical features from rolling sensor windows."""
    features = pd.DataFrame()
    for col in sensor_df.select_dtypes(include=[np.number]).columns:
        features[f"{col}_mean"] = sensor_df[col].rolling(24).mean()
        features[f"{col}_std"] = sensor_df[col].rolling(24).std()
        features[f"{col}_max"] = sensor_df[col].rolling(24).max()
        features[f"{col}_min"] = sensor_df[col].rolling(24).min()
        features[f"{col}_range"] = features[f"{col}_max"] - features[f"{col}_min"]
    return features.dropna()


class PredictiveMaintenanceModel:
    def __init__(self, model_type: str = "gbm", threshold: float = 0.6):
        self.threshold = threshold
        clf = (
            GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.05)
            if model_type == "gbm"
            else RandomForestClassifier(n_estimators=200, max_depth=8)
        )
        self.pipeline = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
        self.feature_names = []

    def fit(self, sensor_df: pd.DataFrame, labels: pd.Series) -> dict:
        X = extract_features(sensor_df)
        y = labels.loc[X.index]
        self.feature_names = X.columns.tolist()
        scores = cross_val_score(self.pipeline, X, y, cv=5, scoring="f1")
        self.pipeline.fit(X, y)
        return {"cv_f1_mean": round(scores.mean(), 3), "cv_f1_std": round(scores.std(), 3)}

    def predict(self, equipment_id: str, sensor_df: pd.DataFrame) -> MaintenanceAlert:
        X = extract_features(sensor_df).tail(1)
        prob = self.pipeline.predict_proba(X)[0][1]
        risk = "HIGH" if prob >= 0.8 else "MEDIUM" if prob >= self.threshold else "LOW"
        action = (
            "Schedule immediate maintenance" if risk == "HIGH"
            else "Monitor closely" if risk == "MEDIUM"
            else "Normal operation"
        )
        importances = self.pipeline.named_steps["clf"].feature_importances_
        top_features = [self.feature_names[i] for i in importances.argsort()[-3:][::-1]]
        return MaintenanceAlert(
            equipment_id=equipment_id,
            failure_probability=round(float(prob), 3),
            risk_level=risk,
            recommended_action=action,
            features_triggered=top_features,
        )
