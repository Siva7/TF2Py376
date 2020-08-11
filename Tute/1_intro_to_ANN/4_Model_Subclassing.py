import tensorflow as tf
import tensorflow.keras as tfk
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

X,Y = fetch_california_housing(return_X_y=True)
X_scaled = StandardScaler.fit_transform(X)
X_train,X_test,Y_train,Y_test = train_test_split(X_scaled,Y)

class CustomModel(tfk.Model):
    def __init__(self,units=20,activation = tfk.activations.relu,**kwargs):
        super.__init__(**kwargs)
        self.hidden = tfk.layers.Dense(units,activation=activation)
        self.output = tfk.layers.Dense(1)

    def __call__(self,input):
        self.hidden(input)

