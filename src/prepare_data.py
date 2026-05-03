import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents = True, exist_ok = True)

aqi = pd.read_csv(RAW_DIR / "sarajevo_hourly.csv")
flights = pd.read_csv(RAW_DIR / "sarajevo_arrivals.csv")

aqi["datetime"] = pd.to_datetime(aqi["datetime"], dayfirst = True)

aqi_columns = [
  "datetime",
  "european_aqi",
  "us_aqi",
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

clean_aqi["smog_risk] = (
  (clean_aqi["european_aqi"] >= 75) &
  (clean_aqi["pm2_5"] >= 25) &
  (clean_aqi["relative_humidity_2m"] >= 80) &
  (clean_aqi["wind_speed_10m"] <= 5)
).astype(int)

clean_aqi.to_csv(PROCESSED_DIR / "clean_aqi.csv", index = False)

flights["scheduled_arrival_datetime"] = pd.to_datetime(
    flights["scheduled_arrival_datetime"]
)

flights["scheduled_hour"] = flights["scheduled_arrival_datetime"].dt.floor("h")
flights["arrival_hour"] = flights["scheduled_arrival_datetime"].dt.hour
flights["day_of_week"] = flights["scheduled_arrival_datetime"].dt.dayofweek
flights["month"] = flights["scheduled_arrival_datetime"].dt.month
flights["is_winter_month"] = flights["month"].isin([11, 12, 1, 2]).astype(int)

# Keep only Landed and Diverted for the first clean model
flights_model = flights[flights["status"].isin(["Landed", "Diverted"])].copy()

flights_model["able_to_land"] = flights_model["status"].map({
    "Landed": 1,
    "Diverted": 0
})

flight_columns = [
    "scheduled_arrival_datetime",
    "scheduled_hour",
    "status",
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

# -------------------------
# Merge flights with AQI
# -------------------------
flight_environment = clean_flights.merge(
    clean_aqi,
    left_on="scheduled_hour",
    right_on="datetime",
    how="inner"
)

flight_environment.to_csv(
    PROCESSED_DIR / "flight_environment_dataset.csv",
    index=False
)

print("Created processed datasets:")
print(PROCESSED_DIR / "clean_aqi.csv")
print(PROCESSED_DIR / "flight_environment_dataset.csv")
print("Rows in final dataset:", len(flight_environment))
  
