"""BTS ingestion: manual date range or automatically select a month.
Lambda handler: lambda_function.lambda_handler if uploaded as lambda_function.py.
"""
from datetime import datetime, timezone


def shift_month(year: int, month: int, months: int) -> tuple[int, int]:
    """Shift YYYY-MM by a signed number of months."""
    index = year * 12 + (month - 1) + months
    return index // 12, index % 12 + 1


def lambda_handler(event, context):
    event = event or {}
    if event.get("mode") == "monthly":
        # E.g. during Nov 2026, lag_months=2 picks Sep 2026.
        # BTS publication can be delayed: monitor Step Functions failures.
        lag = int(event.get("lag_months", 2))
        if not 1 <= lag <= 12:
            raise ValueError("lag_months must be between 1 and 12")
        today = datetime.now(timezone.utc)
        year, month = shift_month(today.year, today.month, -lag)
        start = end = f"{year:04d}-{month:02d}"
    else:
        start = event["start"]
        end = event["end"]

    start_dt = datetime.strptime(start, "%Y-%m")
    end_dt = datetime.strptime(end, "%Y-%m")
    if start_dt > end_dt:
        raise ValueError("Start date must precede end date")
    if start_dt.year < 1987:
        raise ValueError("BTS dataset starts in 1987")

    periods = []
    year, month = start_dt.year, start_dt.month
    while (year, month) <= (end_dt.year, end_dt.month):
        periods.append({"year": year, "month": month})
        year, month = shift_month(year, month, 1)

    return {"start": start, "end": end,
            "total_periods": len(periods), "periods": periods}
