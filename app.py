import streamlit as st
import joblib
from tensorflow.keras.models import load_model
import numpy as np
from datetime import date, timedelta
import requests
from bs4 import BeautifulSoup
from sklearn.linear_model import LinearRegression

def lstm_predictions(next_days):
    def scrap_values():
        today = date.today()
        days_ago = today - timedelta(days=40)
        today_formatted = today.strftime("%d-%b-%Y")
        days_ago_formatted = days_ago.strftime("%d-%b-%Y")
        req = requests.get(
            f"https://agmarknet.gov.in/SearchCmmMkt.aspx?Tx_Commodity=23&Tx_State=MH&Tx_District=14&Tx_Market=172&DateFrom={days_ago_formatted}&DateTo={today_formatted}&Fr_Date={days_ago_formatted}&To_Date={today_formatted}&Tx_Trend=0&Tx_CommodityHead=Onion&Tx_StateHead=Maharashtra&Tx_DistrictHead=Pune&Tx_MarketHead=Pune")
        soup = BeautifulSoup(req.content, "html.parser")
        scrapped_text = soup.get_text()
        scrapped_text = scrapped_text.split("\n")
        scrapped_text = [item for item in scrapped_text if item != '' and len(item) <= 7]

        faq = [index for index, item in enumerate(scrapped_text) if item == 'FAQ']
        modal_prices = []
        for i in range(len(faq)):
            modal_prices.append(scrapped_text[faq[i] + 3])
        modal_prices = modal_prices[1:]
        modal_prices = modal_prices[::-1]
        while (len(modal_prices) != 20):
            modal_prices.pop(0)
        for i in range(len(modal_prices)):
            modal_prices[i] = float(modal_prices[i]) / 100
        return modal_prices

    mod = load_model('onion_lstm_model.h5')
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
    prediction = mod.predict(
        trainX[-n_days_for_prediction:])  # shape = (n, 1) where n is the n_days_for_prediction
    prediction_copies = np.repeat(prediction, 1, axis=-1)
    y_pred_future = sc.inverse_transform(prediction_copies)[:, 0]
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
        prediction = mod.predict(
            trainX[-n_days_for_prediction:])  # shape = (n, 1) where n is the n_days_for_prediction
        prediction_copies = np.repeat(prediction, 1, axis=-1)
        y_pred_future = np.append(y_pred_future, sc.inverse_transform(prediction_copies)[:, 0])

    st.success("Prediction Successful!")
    st.subheader("Predicted Prices:")
    min_price = min(y_pred_future)
    max_price = max(y_pred_future)
    st.line_chart(y_pred_future, use_container_width=True)
    st.write(f"Minimum Price: {min_price:.2f}")
    st.write(f"Maximum Price: {max_price:.2f}")

st.title('Onion Price Prediction')
next_days = st.number_input('Enter the number of days for prediction', min_value=1, step=1, value=60)
if st.button('Predict'):
    lstm_predictions(next_days)
