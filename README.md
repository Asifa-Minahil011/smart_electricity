# Smart Apartment Electricity Consumption Forecasting and Cost Optimization

AI2002 Artificial Intelligence, Semester Project, Fall 2026 (Path A: Software-Based)

Forecasts the next 24 hours of household electricity consumption using
Random Forest (primary model) and XGBoost (comparison model), then recommends
cost-effective times to run flexible appliances under a peak/off-peak tariff.
Delivered as a Streamlit application.

## Team
| Member | Role |
|---|---|
| Asifa Minahil | Data, features, report compilation |
| Mina Hassan | Modeling and evaluation |
| Manahil Aftab | Optimizer and Streamlit app |

## Dataset
UCI Individual Household Electric Power Consumption
(https://archive.ics.uci.edu/dataset/235/individual%2Bhousehold%2Belectric%2Bpower%2Bconsumption).
The raw file (about 130 MB) is NOT in this repo. Download it and place
`household_power_consumption.txt` in the `data/` folder.
The cleaned hourly file `data/hourly.csv` is included.

## Setup
```bash
git clone https://github.com/Asifa-Minahil011/smart_electricity
cd smart_electricity
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # Mac/Linux
pip install -r requirements.txt
```

## How to run
```bash
python -m src.data_prep        # clean data, print statistics, build features
python notebooks/eda.py        # generate EDA figures in reports/figures/
python -m src.model            # train and evaluate models Mina Hassan
streamlit run app.py           # launch the app Manahil Aftab
```

## Project structure
```
data/            raw data (not tracked) and hourly.csv
models/          saved trained models
notebooks/       EDA script
reports/figures/ generated plots
src/data_prep.py load_data(), make_features(), train_test_split_time()
src/model.py     training, evaluation, predict_next_24h()
src/optimizer.py tariff, appliance constraints, scheduling
app.py           Streamlit application
```

## Pipeline
Raw minute data, cleaning, hourly resample, feature engineering, model,
24-hour forecast, scheduler, Streamlit app.

## Data split
Chronological: train Dec 2006 to Dec 2009, test Jan to Nov 2010.

## Limitations
Data is from a single French household (2006 to 2010). The tariff and
appliance constraints are assumptions. Retraining on local meter data would
be needed for real deployment.