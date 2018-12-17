
# python 2.7
# tarefa vogais 21.08.2018

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


class Vogais:

	def __init__(self, X_train, y_train, d, n):
		self.X_train = X_train.values
		self.y_train = y_train.values
		self.w = np.random.sample(len(X_train.columns))
		self.wb = np.random.rand()
		self.d = d
		self.n = n
		self.train()

	def y(self, v):
		if v <= 0:
			return 0
		return 1

	def to_bit(self, y_train):
		if y_train == self.d:
			return 1
		return 0

	def train(self):
		for X_train, y_train in zip(self.X_train, self.y_train):
			v = sum(X_train * self.w) + self.wb
			while self.y(v) != self.to_bit(y_train):
				self.w += self.n * (self.to_bit(y_train) - self.y(v)) * np.array(X_train)
				self.wb += self.n * (self.to_bit(y_train) - self.y(v))
				v = sum(X_train * self.w) + self.wb

	def vogal(self, y):
		if y == 1:
			return [self.d]
		return ['']

	def predict(self, X_test):
		v = sum(X_test * self.w) + self.wb
		return self.vogal(self.y(v))


def main(file):
	accuracy = 0
	for i in range(10):
		vogais = pd.read_csv(file, sep=',', header=None)
		X_train, X_test, y_train, y_test = train_test_split(vogais.drop(0, axis=1), vogais[0], test_size=None)
		perceptrons = {}
		perceptrons['A'] = Vogais(X_train, y_train, 'A', 0.0001)
		perceptrons['E'] = Vogais(X_train, y_train, 'E', 0.0001)
		perceptrons['I'] = Vogais(X_train, y_train, 'I', 0.0001)
		perceptrons['O'] = Vogais(X_train, y_train, 'O', 0.0001)
		perceptrons['U'] = Vogais(X_train, y_train, 'U', 0.0001)
		_predict = pd.DataFrame()
		for test, vogal in zip(X_test.values, y_test.values):
			_predict = _predict.append(pd.DataFrame(perceptrons[vogal].predict(test)), ignore_index=True)
		accuracy += accuracy_score(y_test, _predict)
	return accuracy/10

print main('vogais1.csv')
print main('vogais2.csv')
print main('vogais3.csv')




