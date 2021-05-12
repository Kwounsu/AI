from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split

import pandas as pd


def predict_house_price(epoch, batch):
    df = pd.read_csv('house_price.csv', delim_whitespace=True, header=None)

    data_set = df.values
    X = data_set[:, 0:13]
    Y = data_set[:, 13]

    X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=0.2)

    model = Sequential()
    model.add(Dense(30, input_dim=13, activation='relu'))
    model.add(Dense(6, activation='relu'))
    model.add(Dense(1))

    model.compile(loss='mean_squared_error', optimizer='adam')

    model.fit(X_train, Y_train, epochs=epoch, batch_size=batch)

    Y_prediction = model.predict(X_validation).flatten()

    prediction_rate = 0
    for i in range(10):
        real_price = Y_validation[i]
        predicted_price = Y_prediction[i]
        prediction_rate += abs(real_price - predicted_price)

    return 100 - prediction_rate / 10


print(predict_house_price(200, 30))
