from numpy import loadtxt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

dataset = loadtxt('pima-indians-diabetes.data.csv', delimiter=',')

x_variables = dataset[:, 0:8]
y_variables = dataset[:, 8]

model = Sequential()
# first hidden layer
# input_dim = variables for data
model.add(Dense(12, input_dim=8, activation='relu'))
# second hidden layer
model.add(Dense(8, activation='relu'))
# output layer
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_variables, y_variables, epochs=150, batch_size=10)

_, accuracy = model.evaluate(x_variables, y_variables)
print(f'Accuracy: {accuracy * 100}')