from flask import Flask, request, render_template, send_file
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import io

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
    
    return model, X_test, y_test

def predict_future_price(model, future_dates):
    future_dates_ordinal = np.array([date.toordinal() for date in future_dates]).reshape(-1, 1)
    predictions = model.predict(future_dates_ordinal)
    return predictions

def plot_predictions(data, future_data):
    plt.figure(figsize=(10, 5))
    plt.plot(data['Date'], data['High'], label='Actual High Prices')
    plt.plot(future_data['Date'], future_data['Predicted_High'], label='Predicted High Prices', linestyle='--')
    plt.xlabel('Date')
    plt.ylabel('High Price')
    plt.title('Stock Price Prediction')
    plt.legend()
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    return img

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        ticker = request.form['ticker']
        start_date = request.form['start_date']
        days_periode = int(request.form['days_periode'])
        
        stock_data = get_stock_data(ticker, start_date)
        stock_data = prepare_data(stock_data)
        
        model, X_test, y_test = train_model(stock_data)
        
        future_dates = pd.date_range(stock_data['Date'].max(), periods=days_periode + 1).tolist()
        future_predictions = predict_future_price(model, future_dates)
        
        future_data = pd.DataFrame({
            'Date': future_dates,
            'Predicted_High': future_predictions
        })
        
        plot_img = plot_predictions(stock_data, future_data)
        
        return render_template('result.html', future_data=future_data, ticker=ticker)
    
    return render_template('index.html')

@app.route('/plot.png')
def plot_png():
    ticker = request.args.get('ticker')
    start_date = request.args.get('start_date')
    days_periode = int(request.args.get('days_periode'))
    
    stock_data = get_stock_data(ticker, start_date)
    stock_data = prepare_data(stock_data)
    
    model, X_test, y_test = train_model(stock_data)
    
    future_dates = pd.date_range(stock_data['Date'].max(), periods=days_periode + 1).tolist()
    future_predictions = predict_future_price(model, future_dates)
    
    future_data = pd.DataFrame({
        'Date': future_dates,
        'Predicted_High': future_predictions
    })
    
    img = plot_predictions(stock_data, future_data)
    
    return send_file(img, mimetype='image/png')

if __name__ == "__main__":
    app.run(debug=True)