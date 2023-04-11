from pandas import DataFrame
import numpy as np
from pandas import read_csv, DataFrame, to_datetime

# class AutoRegressive:
#     def __init__(self, p: int, time_series: DataFrame):
#         """
#         Creates a new AutoRegressive model.
#         :param time_series: the time series to fit the model to
#         """
#         self.time_series = time_series
#         self.weights = np.random.rand(p)
        

#     def _sgd_step(self, learning_rate: float):
#         """
#         Perform a single step of stochastic gradient descent on the AutoRegressive model.
#         :param time_series: the time series to fit the model to
#         :param learning_rate: the learning rate to use for the step
#         """
#         pass

#     def fit(self, learning_rate: float, epochs: int):
#         """
#         Fit the AutoRegressive model to the given time series.


#         :param time_series: the time series to fit the model to
#         """
#         # Initialize weights 
#         self.weights = np.random.rand(self.p)

class ARIMA:
    def __init__(self, p: int, d: int, q: int):
        """
        Creates a new ARIMA model.
        :param p: the number of components to consider in the AutoRegressive model
        :param d: the number of times to apply the difference function
        :param q: the number of components to consider in the moving average model.
        """
        self.p = p
        self.d = d
        self.q = q
        self.ar_weights = np.random.rand(p)
        self.ma_weights = np.random.rand(q)

    def fit(self):
        pass

    def initialise_error(self, time_series: DataFrame):
        """
        Create a new column in the table called 'error' and set all the initial values in the column to zero

        :param time_series: the time series dataframe to initialise the error values for
        """
        time_series['error'] = 0.
        return time_series    

    def difference(self, time_series: DataFrame, d: int):
        """
        Compute the difference between the current and previous value of a series.
        Assumes the series is sorted by time, the time column is named 'time' and the series is named 'value'.
        Will produce a new column named 'diff' in the dataframe.

        :param time_series: the time series to compute the difference for
        :param d: the difference parameter
        """
        assert d > 0, 'd must be a non-zero positive integer'

        # Create a new column for the difference, and initialize it with the original series
        time_series['diff'] = time_series['value']

        # Apply the difference function d times
        while d > 0:
            time_series['diff'] = time_series['diff'] - time_series['diff'].shift(1)
            d -= 1
        
        return time_series
    
    def initial_diff_hat(self, time_series: DataFrame, start_step: int):
        """
        Compute the initial value of diff hat from the chosen start point that is passed as a paramater
        Will compute diff_hat by getting the difference from the previous two steps
        Creates a new column called diff_hat and adds the value, as well as computing the error.

        :param time_series: the time series we want to compute diff_hat for
        :param start_step: a value representing which time step we wish to begin computing diff_hat from
        """
        time_series['diff_hat'] = 0.
        one_before = time_series.at[start_step-1, 'diff']
        two_before = time_series.at[start_step-2, 'diff']
        diff_hat = one_before * self.ar_weights[0] + two_before * self.ar_weights[1]
        time_series.at[start_step, 'diff_hat'] = diff_hat
        time_series.at[start_step, 'error'] = time_series.at[start_step, 'diff'] - diff_hat 
        return time_series
    
    def time_step(self, time_series: DataFrame, start_step: int):
        """
        Computes diff_hat from the previous two differences and the error
        Also computes the error from the new value for diff_hat
        Will update the time_series with the new values and return the time series.
        """
        one_before = time_series.at[start_step-1, 'diff']
        two_before = time_series.at[start_step-2, 'diff']
        error = time_series.at[start_step-1, 'error']
        diff_hat = one_before * self.ar_weights[0] + two_before * self.ar_weights[1] + error * self.ma_weights[0]
        time_series.at[start_step, 'diff_hat'] = diff_hat
        time_series.at[start_step, 'error'] = time_series.at[start_step, 'diff'] - diff_hat 
        return time_series

def main():
    arima = ARIMA(2,1,1)
    arima.ar_weights = [0.1, 0.7]
    arima.ma_weights = [0.5]
    #bt_prices = read_csv('data/BT-A.L.csv')
    bt_prices = read_csv('data/dummy_data.csv')
    bt_prices = bt_prices.rename(columns={'Date': 'time', 'Close': 'value'})
    bt_prices = arima.initialise_error(bt_prices)
    bt_prices = arima.difference(bt_prices, 1)
    bt_prices = arima.initial_diff_hat(bt_prices, 4)
    for i in range(5,8):
        bt_prices = arima.time_step(bt_prices, i)
        if (np.isnan(bt_prices.at[i, 'value']) == True):
            bt_prices.at[i, 'value'] = bt_prices.at[i-1, 'value'] + bt_prices.at[i, 'diff_hat']
        bt_prices = arima.difference(bt_prices, 1)
        bt_prices.at[i, 'error'] = round(bt_prices.at[i, 'diff'] - bt_prices.at[i, 'diff_hat'], 5)
        bt_prices.at[i, 'time'] = i

    print(bt_prices)

if __name__ == "__main__":
    main()
