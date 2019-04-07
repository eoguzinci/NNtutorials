import pandas as pd
import numpy as np

class NN(object):
	"""docstring for NN"""
	def __init__(self):
		# seed the random number generator
		np.random.seed(1)
		# init weights
		self.weights = []
		self.bias = []
	
	def sigmoid(self, x, deriv=False):
		if deriv == True:
			return x * (1-x)
		return 1 / (1 + np.exp(-x))

	# forward propagation
	def predict(self, X):
		output = np.dot(X, self.weights) + self.bias
		return self.sigmoid(output)

	def train(self, X, y, iterations):
		dim = X.shape # dimensions of the input
		# assign random weights with values range from -1 to 1 and 0 mean
		self.weights = 2*np.random.random((dim[1],1)) - 1
		self.bias = 2*np.random.random() - 1

		# backpropagation
		for i in range(iterations):
			# for each iteration we predict and calculate the error from ground truth
			output = self.predict(X)
			error = output - y

			# then, we calculate the adjustments to be done to weights
			dw = np.dot(X.T, error * self.sigmoid(output, deriv=True))
			db = np.dot(np.array([1,1,1,1]).T, error * self.sigmoid(output, deriv=True))

			# finally, correct the weights
			self.weights -= dw
			self.bias -= db


if __name__ == "__main__":
	#Loading Data
	# data = pd.read_csv("file.csv")
	# display(len(data))

	# X = data.iloc[:, 0:4].values # features
	# y = data.iloc[:, [4]].values # labels

	X = np.array([[0,0,1],[1,1,1],[1,0,1],[0,1,1]])
	y = np.array([[0,1,1,0]]).T
	num_iter = 1000

	# init neural network
	clf = NN()

	# Train
	clf.train(X, y, num_iter)

	print('Weights after training:')
	print(clf.weights)
	print('Bias after training:')
	print(clf.bias)

	# Test
	test_array = [1,1,0]
	test_label = 1
	print('Predicting:')
	result = clf.predict(test_array)

	print(result)
	if result > 0.5:
		print('Correct')
	else:
		print('False')

