import tensorflow as tf
import pandas as pd
import numpy as np
import pickle

stats = pickle.load(open('model\\v1\\diamond_train_stats', 'rb'))

def norm(x):
    return (x - stats['mean']) / stats['std']


def getFormattedDF(carat, cut, color, clarity, x, y, z):
    """
    Converts raw data to sythetic features required for the model to predict.

    Arguments:\n
        carat {float} : weight of the diamond.
        cut {string} : quality of the cut (Fair, Good, Very Good, Premium, Ideal).
        color {string} : diamond colour, from J (worst) to D (best).
        clarity {string} : a measurement of how clear the diamond is (I1 (worst), SI2, SI1, VS2, VS1, VVS2, VVS1, IF (best)).
        x {float} : length in mm.
        y {float} : width in mm.
        z {float} : depth in mm.

    Returns:\n
        DataFrame : A dataframe containing the normalized sythetic features converted from raw data.
    """
    dfDict = {
        'carat': [],
        'volume': [],
        'cut_Fair': [],
        'cut_Good': [],
        'cut_Ideal': [],
        'cut_Premium': [],
        'cut_Very Good': [],
        'color_D': [],
        'color_E': [],
        'color_F': [],
        'color_G': [],
        'color_H': [],
        'color_I': [],
        'color_J': [],
        'clarity_I1': [],
        'clarity_IF': [],
        'clarity_SI1': [],
        'clarity_SI2': [],
        'clarity_VS1': [],
        'clarity_VS2': [],
        'clarity_VVS1': [],
        'clarity_VVS2': []
    }

    dfDict['carat'].append(carat)
    dfDict['volume'].append(x*y*z)
    dfDict['cut_'+cut].append(1)
    dfDict['color_'+color].append(1)
    dfDict['clarity_'+clarity].append(1)

    for key in dfDict:
        if len(dfDict[key]) == 0:
            dfDict[key].append(0)

    dataDF = pd.DataFrame(dfDict)

    norm_dataDF = norm(dataDF)

    return norm_dataDF
