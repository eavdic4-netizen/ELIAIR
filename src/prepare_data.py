import re
from pathlib import Path

import pandas as pd


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def load_arrivals(path: Path) -> pd.DataFrame:
    """Load the Numbers-exported arrivals CSV."""
    return pd.read_csv(path, sep=";", skiprows=1)


def parse_origin_city(value: object) -> str | None:
    if pd.isna(value):
        return None

    text = str(value).strip()
    match = re.match(r"^(.*?)\s*\(([^/()]+)\s*/\s*([^/()]+)\)\s*$", text)
    if match:
        return match.group(1).strip()

    return text


def parse_origin_iata(value: object) -> str | None:
    if pd.isna(value):
        return None

    match = re.match(r"^.*?\s*\(([^/()]+)\s*/\s*([^/()]+)\)\s*$", str(value).strip())
    if match:
        return match.group(1).strip()

    return None


def parse_time(value: object):
    if pd.isna(value):
        return pd.NaT

    parts = str(value).strip().split(":")
    if len(parts) == 2:
        hours, minutes = parts
        return pd.to_timedelta(f"{int(hours):02d}:{int(minutes):02d}:00")
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return pd.to_timedelta(
            f"{int(hours):02d}:{int(minutes):02d}:{int(float(seconds)):02d}"
        )

    return pd.NaT


def parse_scheduled_date(values: pd.Series) -> pd.Series:
    dates = pd.to_datetime(values, format="%d-%b-%y", errors="coerce")
    missing = dates.isna()

    if missing.any():
        dates.loc[missing] = pd.to_datetime(
            values.loc[missing],
            format="%m/%d/%y %H:%M",
            errors="coerce",
        )

    return dates


# -------------------------
# Clean AQI dataset
# -------------------------
aqi = pd.read_csv(RAW_DIR / "sarajevo_hourly.csv")
aqi["datetime"] = pd.to_datetime(aqi["datetime"], dayfirst=True)

aqi_columns = [
    "datetime",
    "european_aqi",
    "pm10",
    "pm2_5",
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "snowfall",
    "cloud_cover",
    "wind_speed_10m",
    "wind_gusts_10m",
    "weather_code",
]

clean_aqi = aqi[aqi_columns].copy()
clean_aqi["hour"] = clean_aqi["datetime"].dt.hour
clean_aqi["day_of_week"] = clean_aqi["datetime"].dt.dayofweek
clean_aqi["month"] = clean_aqi["datetime"].dt.month

pollution_score = (
    (clean_aqi["european_aqi"] >= 65).astype(int)
    + (clean_aqi["pm2_5"] >= 25).astype(int)
    + (clean_aqi["pm10"] >= 40).astype(int)
)
stagnant_air = (clean_aqi["relative_humidity_2m"] >= 75) & (
    clean_aqi["wind_speed_10m"] <= 7
)
very_polluted_humid_air = (
    ((clean_aqi["european_aqi"] >= 75) | (clean_aqi["pm2_5"] >= 50))
    & (clean_aqi["relative_humidity_2m"] >= 80)
    & (clean_aqi["wind_speed_10m"] <= 8)
)

clean_aqi["smog_risk"] = (
    ((pollution_score >= 2) & stagnant_air) | very_polluted_humid_air
).astype(int)

clean_aqi.to_csv(PROCESSED_DIR / "clean_aqi.csv", index=False)


# -------------------------
# Clean flights dataset
# -------------------------
flights = load_arrivals(RAW_DIR / "sarajevo_arrivals.csv")
flights["flight_status"] = flights["flight_status"].astype(str).str.strip()

scheduled_date = parse_scheduled_date(flights["scheduled_arrival_datetime"])
scheduled_time = flights["scheduled_hour"].apply(parse_time)
flights["scheduled_arrival_datetime"] = scheduled_date + scheduled_time

flights["scheduled_hour"] = flights["scheduled_arrival_datetime"].dt.floor("h")
flights["arrival_hour"] = flights["scheduled_arrival_datetime"].dt.hour
flights["day_of_week"] = flights["scheduled_arrival_datetime"].dt.dayofweek
flights["month"] = flights["scheduled_arrival_datetime"].dt.month
flights["is_winter_month"] = flights["month"].isin([11, 12, 1, 2]).astype(int)
flights["origin_city"] = flights["city"].apply(parse_origin_city)
flights["origin_iata"] = flights["city"].apply(parse_origin_iata)

# Keep Unknown out of the model, but include Cancelled as a non-landing outcome.
flights_model = flights[
    flights["flight_status"].isin(["Landed", "Diverted", "Cancelled"])
].copy()
flights_model["able_to_land"] = flights_model["flight_status"].map(
    {"Landed": 1, "Diverted": 0, "Cancelled": 0}
)

flight_columns = [
    "scheduled_arrival_datetime",
    "scheduled_hour",
    "flight_status",
    "able_to_land",
    "flight_number",
    "callsign",
    "airline",
    "origin_city",
    "origin_iata",
    "arrival_hour",
    "day_of_week",
    "month",
    "is_winter_month",
]

clean_flights = flights_model[flight_columns].copy()
clean_flights = clean_flights.rename(columns={"flight_status": "status"})


# -------------------------
# Merge flights with AQI
# -------------------------
flight_environment = clean_flights.merge(
    clean_aqi,
    left_on="scheduled_hour",
    right_on="datetime",
    how="inner",
)
flight_environment = flight_environment.rename(
    columns={
        "datetime": "aqi_datetime",
        "day_of_week_x": "day_of_week",
        "month_x": "month",
    }
)
flight_environment = flight_environment.drop(columns=["hour", "day_of_week_y", "month_y"])

flight_environment.to_csv(
    PROCESSED_DIR / "flight_environment_dataset.csv",
    index=False,
)

print("Created processed datasets:")
print(PROCESSED_DIR / "clean_aqi.csv")
print(PROCESSED_DIR / "flight_environment_dataset.csv")
print("Rows in clean AQI dataset:", len(clean_aqi))
print("Rows in clean flight dataset:", len(clean_flights))
print("Rows in final merged dataset:", len(flight_environment))
