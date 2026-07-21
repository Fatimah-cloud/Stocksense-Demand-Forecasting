"""Turns a forecast into a recommended purchase order via safety-stock math.

reorder_point = expected demand over lead time + safety stock
safety_stock  = z(service_level) * std-dev of forecast error over lead time
order_qty     = max(0, reorder_point - current_stock)

The forecast's own RMSE (from the backtest metrics) stands in for the daily
demand-error std-dev; assuming day-to-day errors are roughly independent,
the lead-time error std-dev scales with sqrt(lead_time_days).
"""

from typing import List

from scipy.stats import norm


def compute_order_suggestion(
    forecast_points: List[dict],
    current_stock: float,
    lead_time_days: int,
    service_level: float,
    rmse: float,
) -> dict:
    lead_time_forecast = forecast_points[:lead_time_days]
    expected_demand = sum(p["forecast"] for p in lead_time_forecast)

    z = float(norm.ppf(service_level))
    demand_std = rmse * (lead_time_days**0.5)
    safety_stock = max(z * demand_std, 0.0)

    reorder_point = expected_demand + safety_stock
    recommended_qty = max(reorder_point - current_stock, 0.0)

    return {
        "expected_demand_lead_time": round(expected_demand, 2),
        "safety_stock": round(safety_stock, 2),
        "reorder_point": round(reorder_point, 2),
        "recommended_order_qty": round(recommended_qty, 2),
    }
