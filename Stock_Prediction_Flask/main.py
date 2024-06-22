from flask import Flask, request, render_template
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

app = Flask(__name__)

def get_stock_data(ticker, start_date):
   stock_data = yf.download(ticker, start=start_date)
   return stock_data

def prepare_data(data):
    data['Date'] = data.index
    data['Date'] = pd.to_datetime(data['Date'])
    data['Date_ordinal'] = data['Date'].apply(lambda x: x.toordinal())
    return data

def train_model(data):
    X = data[['Date_ordinal']]
    y = data['High']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    return model

def predict_future_price(model, future_dates):
    future_dates_ordinal = np.array([date.toordinal() for date in future_dates]).reshape(-1, 1)
    predictions = model.predict(future_dates_ordinal)
    return predictions

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        ticker = request.form['ticker']
        start_date = request.form['start_date']
        days_periode = int(request.form['days_periode'])
        
        stock_data = get_stock_data(ticker, start_date)
        stock_data = prepare_data(stock_data)
        
        model = train_model(stock_data)
        
        future_dates = pd.date_range(stock_data['Date'].max(), periods=days_periode + 1).tolist()
        future_predictions = predict_future_price(model, future_dates)
        
        future_data = pd.DataFrame({
            'Date': future_dates,
            'Predicted_High': future_predictions
        })
        
        return render_template('result.html', future_data=future_data, ticker=ticker)
    
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)