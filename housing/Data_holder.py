import pandas as pd


class Data_holder:
    def __init__(self):
        self._data = None

    def set_data(self, data):
        self._data = data

    def get_data(self) -> pd.DataFrame:
        return self._data
