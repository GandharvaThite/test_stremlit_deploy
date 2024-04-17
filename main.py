import joblib
import matplotlib.pyplot as plt
import numpy as np
import requests
from datetime import date, timedelta
from bs4 import BeautifulSoup
import pickle
# import pickle_utils

import keras
import tensorflow as tf
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

def lstm_predictions():
    def net_load_model(filename):
        with open(filename, 'rb') as f:
            model = pickle.load(f)
        return model
    def scrap_values():
        today = date.today()
        days_ago = today - timedelta(days=40)
        today_formatted = today.strftime("%d-%b-%Y")
        days_ago_formatted = days_ago.strftime("%d-%b-%Y")
        req = requests.get(f"https://agmarknet.gov.in/SearchCmmMkt.aspx?Tx_Commodity=23&Tx_State=MH&Tx_District=14&Tx_Market=172&DateFrom={days_ago_formatted}&DateTo={today_formatted}&Fr_Date={days_ago_formatted}&To_Date={today_formatted}&Tx_Trend=0&Tx_CommodityHead=Onion&Tx_StateHead=Maharashtra&Tx_DistrictHead=Pune&Tx_MarketHead=Pune")
        soup = BeautifulSoup(req.content, "html.parser")
        scrapped_text = soup.get_text()
        scrapped_text = scrapped_text.split("\n")
        scrapped_text = [item for item in scrapped_text if item != ''and len(item) <= 7]

        faq = [index for index, item in enumerate(scrapped_text) if item == 'FAQ']
        modal_prices = []
        for i in range(len(faq)):
            modal_prices.append(scrapped_text[faq[i]+3])
        modal_prices = modal_prices[1:]
        modal_prices = modal_prices[::-1]
        while(len(modal_prices) != 20):
            modal_prices.pop(0)
        for i in range(len(modal_prices)):
            modal_prices[i] = float(modal_prices[i])/100
        return modal_prices

    def map_findings(y_pred_future):
        regression_model = LinearRegression()
        X = np.arange(len(y_pred_future)).reshape(-1, 1)
        regression_model.fit(X, y_pred_future)
        trendline = regression_model.predict(X)
        plt.figure(figsize=(20, 6))
        plt.plot(y_pred_future, marker='o', linestyle='-', color='blue', label='Prices', markersize=4)
        plt.plot(trendline, linestyle='--', color='black', label='Trendline')
        for i in range(1, len(y_pred_future)):
            if y_pred_future[i] < y_pred_future[i-1]:
                plt.plot([i-1, i], [y_pred_future[i-1], y_pred_future[i]], linestyle='-', color='red', linewidth=2)
            else:
                plt.plot([i-1, i], [y_pred_future[i-1], y_pred_future[i]], linestyle='-', color='green', linewidth=2)
        plt.xlabel('Index', fontsize=12)
        plt.ylabel('Price', fontsize=12)
        plt.title('Price Movement with Trendline and Candlestick', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        plt.show()

    loaded_model_json = joblib.load('model.joblib')

    # Parse the JSON string back into a Keras model
    # ("c.h5")
    mod = joblib.load('lstm_onion_model.pkl')
    sc = joblib.load('onion_scaler.pkl')
    X = scrap_values()
    X = np.asarray(X)
    df_for_training = X.astype(float)
    n_rows = len(df_for_training)
    df_for_training = df_for_training.reshape(n_rows, -1)
    scaler = sc
    scaler = scaler.fit(df_for_training)
    df_for_training_scaled = scaler.transform(df_for_training)
    trainX = []
    trainY = []

    n_future = 1  # Number of days we want to look into the future based on the past days.
    n_past = 14  # Number of past days we want to use to predict the future.
    n_days_for_prediction = 6

    for i in range(n_past, len(df_for_training_scaled) - n_future + 1):
        trainX.append(df_for_training_scaled[i - n_past:i, 0:df_for_training.shape[1]])
        trainY.append(df_for_training_scaled[i + n_future - 1:i + n_future, 0])
    trainX, trainY = np.array(trainX), np.array(trainY)

    prediction = mod.predict(trainX[-n_days_for_prediction:])  # shape = (n, 1) where n is the n_days_for_prediction
    prediction_copies = np.repeat(prediction, 1, axis=-1)
    y_pred_future = sc.inverse_transform(prediction_copies)[:, 0]

    next_days = 60
    for i in range(0, int(next_days / n_days_for_prediction)):
        for j in range(0, 6):
            X = np.delete(X, 0)
        for j in range(len(y_pred_future)):
            X = np.append(X, y_pred_future[i])
        df_for_training = X.astype(float)
        n_rows = len(df_for_training)
        df_for_training = df_for_training.reshape(n_rows, -1)
        scaler = sc
        scaler = scaler.fit(df_for_training)
        df_for_training_scaled = scaler.transform(df_for_training)
        trainX = []
        trainY = []

        for i in range(n_past, len(df_for_training_scaled) - n_future + 1):
            trainX.append(df_for_training_scaled[i - n_past:i, 0:df_for_training.shape[1]])
            trainY.append(df_for_training_scaled[i + n_future - 1:i + n_future, 0])
        trainX, trainY = np.array(trainX), np.array(trainY)

        prediction = mod.predict(trainX[-n_days_for_prediction:])  # shape = (n, 1) where n is the n_days_for_prediction
        prediction_copies = np.repeat(prediction, 1, axis=-1)
        y_pred_future = np.append(y_pred_future, sc.inverse_transform(prediction_copies)[:, 0])

    print("Length of prediction array:", len(y_pred_future))
    print("Predicted prices:", y_pred_future)
    map_findings(y_pred_future)
    return y_pred_future

vals = lstm_predictions()
