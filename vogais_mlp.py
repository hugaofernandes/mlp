
# python 2.7
# tarefa 12.09.2018

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from  sklearn.neural_network  import  MLPClassifier


def main(file, n):
	_predict = 0
	for i in range(10):
		vogais = pd.read_csv(file, sep=',', header=None)
		X_train, X_test, y_train, y_test = train_test_split(vogais.drop(0, axis=1), vogais[0], test_size=None)
		mlp = MLPClassifier(hidden_layer_sizes=(n))
		mlp.fit(X_train, y_train)
		_predict += accuracy_score(y_test, mlp.predict(X_test))
	return _predict/10


print main('vogais1.csv', 5)
print main('vogais2.csv', 5)
print main('vogais3.csv', 5)

print main('vogais1.csv', 10)
print main('vogais2.csv', 10)
print main('vogais3.csv', 10)

print main('vogais1.csv', 100)
print main('vogais2.csv', 100)
print main('vogais3.csv', 100)

