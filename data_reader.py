import os
import numpy as np
import random
import pandas as pd
from CustomImputer import Impute

class Reader(object):
    def __init__(self, dataset_dir, listfile=None):
        self._dataset_dir = dataset_dir
        self._current_index = 0
        if listfile is None:
            listfile_path = os.path.join(dataset_dir, "listfile.csv")
        else:
            listfile_path = listfile
        self._data = pd.read_csv(listfile_path)
 
    def get_number_of_examples(self):
        return len(self._data)

    def random_shuffle(self, seed=None):
        if (seed is not None):
            random.seed(seed)
        random.shuffle(self._data)

    def read_example(self, index):
        raise NotImplementedError()

    def read_next(self):
        to_read_index = self._current_index
        self._current_index += 1
        if (self._current_index == self.get_number_of_examples()):
            self._current_index = 0
        return self.read_example(to_read_index)


class InHospitalMortalityReader(Reader):
    def __init__(self, dataset_dir, listfile=None, period_length=48.0):
        """ 
        :param period_length: Length of the period (in hours) from which the prediction is done.
        """
        Reader.__init__(self, dataset_dir, listfile)
        self._period_length = period_length

    
        
    def _load_data(self, discretizer, small_part=False):
        N = self.get_number_of_examples()
        if small_part:
            N = 1
        data = []
        for i in range(N):
            nextEpisode = self.read_next()
            data.append(nextEpisode)
            if i % 100 == 0:
                print("processed {} / {}\r".format(i+1, N))

        return data
    

            
    def _read_timeseries(self, ts_filename):
        try:
            return pd.read_csv(os.path.join(self._dataset_dir, ts_filename))
        except: 
            print(ts_filename + " could not be loaded")
            return null

    def read_example(self, index):
        if (index < 0 or index >= len(self._data)):
            raise ValueError("Index must be from 0 (inclusive) to number of lines (exclusive).")

        X = self._read_timeseries(self._data["stay"][index])
        hours = X['Hours']
        X = Impute(X)
        X['Hours'] = hours
        X['Mortality'] = self._data["y_true"][index]
        X['Episode'] = self._data["stay"][index]

        return X
