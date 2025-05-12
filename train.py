import yfinance as yf
from datetime import date 
from prophet import Prophet
from sklearn.metrics import mean_squared_error
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
import mlflow
import mlflow.prophet
import joblib

TICKER = 'BBAS3.SA'
START_DATE = '2020-01-01'
END_DATE = date.today().strftime('%Y-%m-%d')

def download_data(ticker: str, start_date: str, end_date: str):
    """
    Retorna os dados históricos da ação.
    """

    stock_data = yf.Ticker(ticker)

    stock_data = stock_data.history(period='1d', start=start_date, end=end_date)

    stock_data = stock_data.reset_index('Date')

    stock_data['Date'] = stock_data['Date'].apply(lambda x: x.replace(tzinfo=None))

    stock_data = stock_data[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})

    return stock_data


def evaluate_model(y_true, y_pred):
    """
    Retorna métricas de avaliação do modelo: MAE, MAPE, RSE e RMSE.
    """
    
    mae = mean_absolute_error(y_true, y_pred)

    mape = mean_absolute_percentage_error(y_true, y_pred)

    mse = mean_squared_error(y_true, y_pred)

    rmse = root_mean_squared_error(y_true, y_pred)

    return mae, mape, mse, rmse


def train_model():
    """
    Treina e salva o modelo.
    """

    stock_data = download_data(TICKER, START_DATE, END_DATE)

    train_data = stock_data.sample(frac=0.8, random_state=0)
    test_data = stock_data.drop(train_data.index)

    mlflow.set_experiment("previsao-acoes-prophet")

    with mlflow.start_run():

        model = Prophet(daily_seasonality=True)

        model.fit(train_data)

        future = test_data[['ds']]

        forecast = model.predict(future)

        y_pred = forecast['yhat']

        y_true = test_data['y']

        input_example = future.head(10)

        mae, mape, mse, rmse = evaluate_model(y_true, y_pred)

        mlflow.log_param('ticker', TICKER)
        mlflow.log_param('model', 'Prophet')
        mlflow.log_metric('mae', mae)
        mlflow.log_metric('mape', mape)
        mlflow.log_metric('mse', mse)
        mlflow.log_metric('rmse', rmse)

        mlflow.prophet.log_model(model, artifact_path='model', input_example=input_example)

        joblib.dump(model, 'model/prophet_model.pkl')


if __name__ == "__main__":
    train_model()