# Ignore warnings :
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import pickle

from processData import getFormattedDF

model = keras.models.load_model('model\\v1\\DiamondPricePredictor_model_loss-364475.h5')

def predictPrice(carat, cut, color, clarity, x, y, z):
    """
    Predict the price of the diamond in US dollars

    Arguments:\n
        carat {float} : weight of the diamond.
        cut {string} : quality of the cut (Fair, Good, Very Good, Premium, Ideal).
        color {string} : diamond colour, from J (worst) to D (best).
        clarity {string} : a measurement of how clear the diamond is (I1 (worst), SI2, SI1, VS2, VS1, VVS2, VVS1, IF (best)).
        x {float} : length in mm.
        y {float} : width in mm.
        z {float} : depth in mm.

    Returns:\n
        Float : The price of the diamond
    """
    dataDF = getFormattedDF(carat, cut, color, clarity, x, y, z)
    price = model.predict(dataDF)
    return price[0][0]


# print(predictPrice(carat=2.29, cut='Premium', color='I', clarity='VS2', x=8.5, y=8.47, z=5.16))