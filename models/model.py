import os
import pickle
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

class Model(object):
    """docstring for Model."""

    def __init__(self):
        super(Model, self).__init__()
        self.scaler = StandardScaler()
        self.svm = SVC(kernel = 'rbf', random_state = 0)
        self.cleaner = SimpleImputer(missing_values = np.nan, strategy = 'mean')
        self.load_datasets()

    def load_datasets(self):
        self.data = {
            'train': { 'x': None, 'y': None },
            'test': { 'x': None, 'y': None }
        }

        # Training data
        data = pd.read_csv(os.path.join(os.getcwd(), 'dataset', 'train.csv'))
        self.data['train']['x'] = data.iloc[ :, :-1].values
        self.data['train']['y'] = data.iloc[ :, 13].values

        # Testing data
        data = pd.read_csv(os.path.join(os.getcwd(), 'dataset', 'test.csv'))
        self.data['test']['x'] = data.iloc[:, :].values

    def clean_data(self, data, is_training = False):
        fitted = self.cleaner.fit(data[:, :])
        if(is_training):
            self.cleaner = fitted
        return fitted.transform(data[:, :])

    def train(self):
        cX = self.clean_data(self.data['train']['x'], True)
        X = self.scaler.fit_transform(cX)

        self.svm.fit(X, self.data['train']['y'])

    def predict(self, data = []):
        if(not len(data)):
            print('Data not found! Testing with default dataset!')
            data = self.data['test']['x']

        num_points = len(data)
        cX = self.clean_data(np.concatenate([data, self.data['train']['x']]))
        X = self.scaler.fit_transform(cX)

        return self.svm.predict(X)[:num_points]

    def save(self):
        with open('model.pkl', 'wb') as target:
            pickle.dump(self, target)
            target.close()
