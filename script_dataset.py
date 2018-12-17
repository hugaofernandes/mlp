

# python 2.7

import pandas as pd
import numpy as np


def dataset(n, rate, bitchange, file):
	vogais = pd.DataFrame()
	for v in [['A',0,1,0,1,0,1,1,1,1,1,0,1,1,0,1], ['E',1,1,1,1,0,0,1,1,1,1,0,0,1,1,1],	['I',0,1,0,0,1,0,0,1,0,0,1,0,0,1,0], ['O',0,1,0,1,0,1,1,0,1,1,0,1,0,1,0], ['U',1,0,1,1,0,1,1,0,1,1,0,1,0,1,0]]:
		for i in range(n/5):
			vogal = v
			if rate > np.random.rand():
				for j in range(bitchange):
					index = np.random.randint(1, len(vogal))
					if vogal[index] == 1:
						vogal[index] = 0
					else:
						vogal[index] = 1
			vogais = vogais.append(pd.DataFrame([vogal]), ignore_index=True)
	vogais.to_csv(file, sep=',', encoding='utf-8', header=None, index=None)
	return vogais


dataset(1000, 0.1, 1, 'vogais1.csv')
dataset(1000, 0.1, 2, 'vogais2.csv')
dataset(1000, 0.1, 3, 'vogais3.csv')

