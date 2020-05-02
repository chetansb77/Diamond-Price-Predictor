# Diamond-Price-Predictor
A Linear Regression model that predicts the price of the diamond.</br>
The model is trained on a dataset containing the attributes of almost 54,000 diamonds.

### List of features
* price: price in US dollars (\$326--\$18,823)
* carat: weight of the diamond (0.2--5.01)\n
* cut: quality of the cut (Fair, Good, Very Good, Premium, Ideal)
* color: diamond colour, from J (worst) to D (best)
* clarity: a measurement of how clear the diamond is (I1 (worst), SI2, SI1, VS2, VS1, VVS2, VVS1, IF (best))
* x: length in mm (0--10.74)
* y: width in mm (0--58.9)
* z: depth in mm (0--31.8)
* depth: total depth percentage = z / mean(x, y) = 2 * z / (x + y) (43--79)
* table: width of top of diamond relative to widest point (43--95)</br>

The Diamonds.ipynb has a detailed explaination and code for exploratory data analysis and model creation
using Tensorflow.

## Getting Started

### Prerequisites
You should first install all the dependency libraries by running the following command
```
pip install -r requirements.txt
```

## How to Predict
```
from predict import predictPrice

# Example
predictPrice(carat=2.29, cut='Premium', color='I', clarity='VS2', x=8.5, y=8.47, z=5.16)

# returns float value of price
```
