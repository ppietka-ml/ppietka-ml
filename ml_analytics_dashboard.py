"""
ML analytics dashboard backend for business intelligence reporting.
Aggregates model outputs and KPIs into structured API responses for visualization.
"""

from dataclasses import dataclass
from datetime import datetime
import pandas as pd
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel


@dataclass
class KPISnapshot:
    period: str
    revenue_forecast: float
    forecast_accuracy: float
    anomalies_detected: int
    top_performing_skus: list
    alert_count: int


class DashboardRequest(BaseModel):
    start_date: str
    end_date: str
    granularity: str = "weekly"


class MLAnalyticsDashboard:
    def __init__(self):
        self.app = FastAPI(title="ML Analytics Dashboard API")
        self.kpi_cache = {}
        self._register_routes()

    def _register_routes(self):
        @self.app.post("/kpi-snapshot")
        def get_kpi_snapshot(req: DashboardRequest):
            return self.compute_snapshot(req.start_date, req.end_date, req.granularity)

        @self.app.get("/model-performance")
        def get_model_performance():
            return self.summarize_model_performance()

    def compute_snapshot(self, start: str, end: str, granularity: str) -> dict:
        dates = pd.date_range(start=start, end=end, freq="W" if granularity == "weekly" else "D")
        revenue = np.random.normal(50000, 5000, len(dates))
        accuracy = np.clip(np.random.normal(0.88, 0.03, len(dates)), 0.75, 0.99)
        return {
            "period": f"{start} to {end}",
            "revenue_forecast_total": round(float(revenue.sum()), 2),
            "avg_forecast_accuracy": round(float(accuracy.mean()), 3),
            "anomalies_detected": int(np.random.poisson(3)),
            "data_points": len(dates),
            "generated_at": datetime.utcnow().isoformat(),
        }

    def summarize_model_performance(self) -> dict:
        return {
            "predictive_maintenance": {"mape": 0.087, "precision": 0.91, "recall": 0.88},
            "sales_forecasting": {"mape": 0.112, "rmse": 423.5},
            "anomaly_detection": {"f1": 0.86, "false_positive_rate": 0.04},
            "last_updated": datetime.utcnow().isoformat(),
        }

    def export_report(self, snapshot: dict) -> pd.DataFrame:
        return pd.DataFrame([snapshot])
