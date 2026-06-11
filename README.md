# ELIAIR - PREDICTING AIR QUALITY AND LANDING SAFETY VIA LSTM-XGBOOST FUSION

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-blue?logo=python"/>
  <img src="https://img.shields.io/badge/Platform-Google%20Colab-orange?logo=googlecolab"/>
  <img src="https://img.shields.io/badge/Data-AQI%20%2B%20Flight-lightgrey"/>
  <img src="https://img.shields.io/badge/Model-LSTM-teal"/>
  <img src="https://img.shields.io/badge/Model-XGBoost-teal"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow"/>
</p>


Sarajevo sits in a valley surrounded by mountains, the geography that traps pollutants and fog with striking regularity. Each winter the city ranks among Europe's most polluted cities with PM2.5 and PM10 levels. For pilots approaching Sarajevo International Airport this is a potential operational challenge.

Existing systems treat air quality and flight safety as separate domains. ELIAIR connects them by creating a modular system (LSTM, XGBoost) to forecast pollution trajectories and make actionable landing safety assessments before conditions deteriorate beyond safe limits.


## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [ELIAIR Flight Risk Simulator](#eliair-flight-risk-simulator)
- [Usage](#usage)
- [Methodology](#methodology)
- [Results](#results)
- [Limitations & Future Work](#limitations--future-work)
- [Conclusion](#conclusion)
- [Contributing](#contributing)
- [License](#license)
## Poster

<img width="2200" height="3400" alt="ELIAIR POSTER FIXED-1" src="https://github.com/user-attachments/assets/b36eb670-2102-48bc-b242-d72472619498" />


## Overview
 
Air quality in Sarajevo is among the worst in Europe during winter months, driven by topographic inversions, coal heating, and traffic emissions. ELIAIR bridges two domains that are rarely combined - environmental science and aviation operations - to produce a data-driven, hour-ahead safety signal for flight landings at Sarajevo International Airport.
 
The system consists of two independently trained models:
 
- **LSTM** - forecasts the next hour's European AQI value from a 24-hour window of pollution and meteorological data
- **XGBoost** - classifies whether a flight will successfully land, given environmental conditions, airline, origin, and seasonal context
The two models currently operate in parallel rather than in sequence — both consume the same underlying AQI dataset, but do not interact at inference time. Connecting them so that the LSTM's forecast feeds directly into XGBoost is an identified direction for future work.
 
Data is sourced from [Open-Meteo](https://open-meteo.com/) (meteorological), [OpenAQ](https://openaq.org/) (air quality), and [Flightera](https://www.flightera.net/) (flight operations), covering March 2022-March 2026.
 
## Installation
 
No local installation is required. The notebooks are designed to run on [Google Colab](https://colab.research.google.com/), which provides free GPU access and all necessary dependencies pre-installed.
 
1. Clone or download the repository
2. Upload the notebooks to Google Colab
3. When prompted, upload the required CSV files from the `data/` folder
The following packages are used and are all available by default in Colab:
```
torch, scikit-learn, xgboost, pandas, numpy, matplotlib, seaborn
```

## ELIAIR Flight Risk Simulator

The zipped folder contains the ELIAIR website demo. It runs locally on your own
computer, then opens in your browser.

The website estimates Sarajevo flight landing/disruption risk from flight
details, weather, and air-quality conditions.

**Quick Start**

1. Unzip `eliair-flight-risk-simulator.zip`.
2. Open the unzipped `eliair-flight-risk-simulator` folder.
3. Start the app using the instructions for your operating system below.
4. Keep the Terminal/command window open while using the website.

The website will open automatically if everything works.

If it does not open automatically, copy the URL printed in the Terminal/command
window. It will look similar to this:

```text
http://127.0.0.1:8000/eliair-flight-risk-simulator
```

The port may be `8001`, `8002`, or another number if `8000` is already busy.
Always use the exact URL printed by the launcher.

**Steps for windows**

1. Open the `eliair-flight-risk-simulator` folder.
2. Double-click `start_windows.bat`.
3. Wait while packages install.
4. The browser should open automatically.

If the browser does not open, copy the URL from the black command window and
paste it into your browser.

**Steps for macOS**

Recommended method:

1. Open Terminal.
2. Type `cd ` with a space after it.
3. Drag the `eliair-flight-risk-simulator` folder into Terminal.
4. Press Enter.
5. Run this command:

```bash
python3 start.py
```

Further details are provided within the zip file.
 
## Usage
 
**LSTM - AQI Forecasting**
 
Open and run `notebooks/lstm_aqi_forecasting.ipynb`. The notebook will train the model and save the best checkpoint to `lstm_outputs/lstm_aqi_best.pt`.
 
**XGBoost - Landing Safety Classification**
 
Open and run `notebooks/xgboost_landing_safety.ipynb`. Both CSV files must be available before running. The notebook will merge the datasets, train the classifier, and output evaluation metrics and feature importance plots.
 
**Project structure:**
```
ELIAIR/
├── data/
│   ├── processed/
│   │   ├── clean_aqi.csv
│   │   └── flight_environment_dataset.csv
│   └── raw/
│       ├── sarajevo_arrivals.csv
│       └── sarajevo_hourly.csv
├── graphs/
│   ├── lstm loss.png
│   ├── lstm prediction.png
│   ├── lstm r2.png
│   ├── xgboost features.png
│   └── xgboost matrix.png
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_lstm_aqi_prediction.ipynb
│   └── 03_xgboost_landing_prediction.ipynb
├── src/
│   └── prepare_data.py
├── .gitignore
├── ELIAIR PROJECT POSTER.pdf
├── LICENSE
├── README.md
├── eliair-flight-risk-simulator.zip
└── requirements.txt
```
 
## Methodology
 
### LSTM - AQI Forecasting
 
The LSTM is a two-layer network (128 hidden units, dropout 0.2) trained on 14 input features including European AQI, PM2.5, PM10, temperature, humidity, wind speed, wind gusts, precipitation, snowfall, cloud cover, weather code, and calendar features (hour, day of week, month). It takes a rolling 24-hour window and predicts the AQI one hour ahead.
 
Data is split chronologically — 70% train, 15% validation, 15% test — with MinMaxScaler fit only on training data to prevent leakage. Training uses the Adam optimizer with MSE loss, gradient clipping, a `ReduceLROnPlateau` scheduler, and early stopping (patience = 10).
 
### XGBoost - Landing Safety Classification
 
The XGBoost classifier predicts a binary outcome: whether a flight lands successfully. Flight records are merged with hourly AQI data on scheduled arrival time, producing a combined feature set of 22 variables. Categorical features (airline, origin city, IATA code) are one-hot encoded. Class imbalance — only 320 "did not land" events out of 27,626 flights — is addressed via sample weighting. The model runs 300 boosting rounds with max depth 4 and learning rate 0.05.
 
## Results
 
| Model | Metric | Value |
|---|---|---|
| LSTM | MAE | 0.957 |
| LSTM | RMSE | 1.270 |
| LSTM | R² | 0.9964 |
| XGBoost | Accuracy | 92.15% |
| XGBoost | ROC AUC | 0.7857 |
| XGBoost | F1 (landed) | 0.96 |
| XGBoost | F1 (did not land) | 0.12 |
 
The LSTM captures AQI dynamics with high accuracy. The XGBoost classifier performs well on the majority class but struggles with the rare unsafe-landing events - a known challenge with heavily imbalanced real-world aviation data.

### LSTM Results
 
<p align="center">
  <img src="graphs/lstm%20loss.png" width="700"/>
  <br><em>Training vs validation loss curve</em>
</p>
<p align="center">
  <img src="graphs/lstm%20prediction.png" width="700"/>
  <br><em>Actual vs predicted AQI values</em>
</p>
<p align="center">
  <img src="graphs/lstm%20r2.png" width="450"/>
  <br><em>R² scatter plot</em>
</p>

### XGBoost Results
 
<p align="center">
  <img src="graphs/xgboost%20features.png" width="700"/>
  <br><em>Top feature importances by gain</em>
</p>
<p align="center">
  <img src="graphs/xgboost%20matrix.png" width="450"/>
  <br><em>Confusion matrix</em>
</p>

 
## Limitations & Future Work
 
- **Feature engineering** - the two models are currently trained independently. A planned next step is connecting them so that the LSTM's AQI forecast becomes a direct input feature to the XGBoost classifier, creating a true end-to-end pipeline.
- **Improving the classifier** - the severe class imbalance (320 unsafe landings out of 27,626 flights) limits the model's ability to reliably detect dangerous conditions. Future work includes oversampling techniques like SMOTE, refining the "did not land" label to exclude outcomes unrelated to environmental conditions such as mechanical issues or scheduling, and lowering the classification threshold to recover recall on unsafe landings.
- **Expanding to all Bosnian airports** - the model is currently scoped to Sarajevo. A natural extension is adapting it for Tuzla, Banja Luka, and Mostar, each of which has distinct topographic and pollution profiles worth modeling separately.
- **Better flight data** - current flight data is sourced from public tracking sites. Obtaining historical landing records directly from airports would improve data quality and also allow exploration of what actually happens to flights that fail to land at their intended destination — whether they divert, return, or hold.

## Conclusion

ELIAIR shows that air quality forecasting and flight safety assessment can be meaningfully connected through a hybrid machine learning pipeline. The LSTM component successfully captured the dynamics of Sarajevo’s pollution patterns and produced highly-accurate hour-ahead AQI predictions. The XGBoost classifier then translated environmental and operational context into landing safety assessments, demonstrating that the meteorological conditions, seasonality and route characteristics, all combined, carry predictive signals for flight outcomes. The two models currently operate in parallel, which will be changed into operating in sequence in the future. The classifier’s difficulties with unsafe landing events also highlights the challenge of working with real-world aviation data. 
Overall, ELIAIR lays a solid foundation for AI-assisted environmental risk assessment in aviation, with its design making it a promising basis for further development toward a deployable decision-support tool for airports.


## Contributing
 
Contributions are welcome. Please open an issue first to discuss what you would like to change, then submit a pull request.
 
## License
 
MIT License

## Team members:
| Name | GitHub |
|---|---|
| Ena Avdić | [@eavdic4-netizen](https://github.com/eavdic4-netizen) |
| Iman Duratbegović | [@imoniia](https://github.com/imoniia) |
| Lejla Lolić | [@lejla-lolic](https://github.com/lejla-lolic) |
| Ajna Vegara | [@avegara1-spec](https://github.com/avegara1-spec) |

