import numpy as np
from sklearn.preprocessing import LabelEncoder,StandardScaler
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras as tfk
from sklearn.model_selection import train_test_split
import seaborn as sns
pd.set_option('display.max_columns',None)
train_file_path = r"Kaggle\1_Titanic\train.csv"
test_file_path = r'Kaggle\1_Titanic\test.csv'
# train_file_path = r"train.csv"
# test_file_path = r'test.csv'


train_raw_df = pd.read_csv(train_file_path)
test_raw_df = pd.read_csv(test_file_path)

def encode_data(df):
    df['Sex_encoded'] = LabelEncoder.fit_transform(LabelEncoder(), df[['Sex']])
    df["Age_encoded"] = df["Age"].fillna(np.mean(df["Age"]))
    df['Embarked_encoded'] = LabelEncoder.fit_transform(LabelEncoder(), df[['Embarked']].astype(str))
    df["Age_encoded"] = StandardScaler.fit_transform(StandardScaler(),df[["Age_encoded"]])
    df["Fare_encoded"] = StandardScaler.fit_transform(StandardScaler(),df[["Fare"]])
    return df

out_columns = ["PassengerId","Pclass","Sex_encoded","Age_encoded","SibSp","Parch","Fare_encoded","Embarked_encoded","Survived"]
encoded_train_data = encode_data(train_raw_df)[out_columns]
encoded_test_data = encode_data(test_raw_df)[out_columns[:-1]]

# sns.pairplot(encoded_train_data,hue='Survived')

input_layer = tfk.layers.Input(shape=4)
first_dense_layer = tfk.layers.Dense(10,activation=tfk.activations.tanh)(input_layer)
second_dense_layer = tfk.layers.Dense(10, activation=tfk.activations.tanh)(first_dense_layer)

second_dense_layer = tfk.layers.Dense(10, activation=tfk.activations.tanh)(second_dense_layer)
second_dense_layer = tfk.layers.Dense(10, activation=tfk.activations.tanh)(second_dense_layer)
second_dense_layer = tfk.layers.Dense(10, activation=tfk.activations.tanh)(second_dense_layer)

second_dense_layer = tfk.layers.Dropout(0.6)(second_dense_layer)


linear_input =  tfk.layers.Input(shape=3)
linear_dense_layer_one = tfk.layers.Dense(10,activation=tfk.activations.linear)(linear_input)
linear_dense_layer_two = tfk.layers.Dense(10, activation=tfk.activations.linear)(linear_dense_layer_one)

linear_dense_layer_two = tfk.layers.Dense(10, activation=tfk.activations.linear)(linear_dense_layer_two)
linear_dense_layer_two = tfk.layers.Dense(10, activation=tfk.activations.linear)(linear_dense_layer_two)

linear_drop_out_layer = tfk.layers.Dropout(0.3)(linear_dense_layer_two)

combine_linear_and_non_linear = tfk.layers.concatenate([second_dense_layer,linear_drop_out_layer])
output = tfk.layers.Dense(2,tfk.activations.softmax,kernel_initializer='normal', bias_initializer='zeros')(combine_linear_and_non_linear)

model = tfk.Model(inputs=[input_layer,linear_input],outputs=[output])

model.summary()

X = encoded_train_data[out_columns[:-1]]
Y = encoded_train_data[out_columns[-1]]
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.1)

X_train_tuple = (X_train[["Age_encoded","SibSp","Parch","Embarked_encoded"]],X_train[["Pclass","Sex_encoded","Fare_encoded"]])
X_test_tuple = (X_test[["Age_encoded","SibSp","Parch","Embarked_encoded"]],X_test[["Pclass","Sex_encoded","Fare_encoded"]])
encoded_test_data_tuple = (encoded_test_data[["Age_encoded","SibSp","Parch","Embarked_encoded"]],encoded_test_data[["Pclass","Sex_encoded","Fare_encoded"]])


optimizer = tfk.optimizers.Adam(0.001)
loss_func = tfk.losses.sparse_categorical_crossentropy
model.compile(optimizer=optimizer,loss=[loss_func],metrics=["accuracy"])
training_detail = model.fit(x=X_train_tuple, y=Y_train, verbose=1,epochs=500,
                            batch_size=20,callbacks=tf.keras.callbacks.EarlyStopping(patience=200,restore_best_weights=True,monitor="accuracy",min_delta=0.01))




train_res =np.argmax(model.predict(X_train),axis=1) == Y_train
values = train_res.value_counts()

accuracy = values[True]/(values[True]+values[False])
model.summary()

model.evaluate(X_train_tuple,Y_train)
model.evaluate(X_test_tuple,Y_test)



training_df=pd.DataFrame(training_detail.history)
training_df.plot()
plt.show()
final_predections = np.argmax(model.predict(encoded_test_data_tuple),axis=1)

result_df = pd.concat([encoded_test_data["PassengerId"],pd.Series(final_predections)],axis=1)
result_df.columns = ["PassengerId","Survived"]
result_df.to_csv(r'Kaggle\1_Titanic\prediction.csv',index=False)