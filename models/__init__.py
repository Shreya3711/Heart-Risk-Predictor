import os
import sys
import pickle
import numpy as np

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from model import Model

def get_model():
    with open(os.path.join(os.getcwd(), 'model.pkl'), 'rb') as source:
        model = pickle.load(source)
        source.close()
    return model

def predict(data = []):
    data = np.array([ data ])
    # use prediction over here
    return get_model().predict(data if len(data[0]) else [])

model = Model()
model.train()
model.save()
