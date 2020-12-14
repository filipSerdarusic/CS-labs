from sklearn import svm
import numpy as np
from data import *


class KSVMWrap():

	def __init__(self, X, Y_, c=1, gamma='auto'):
				
		self.svm = svm.SVC(C=c, gamma=gamma)
		self.svm.fit(X, Y_)

	def predict(self, X):
		return self.svm.predict(X)

	def get_scores(self, X):
		Y = self.predict(X)
		return self.svm.score(X,Y)
	
	def get_params(self):
		return self.svm.get_params()

	def support(self):
		return self.svm.support_


if __name__ == "__main__":

	#Initialize random number generators
	np.random.seed(100)

	#Instatiate data X and labels Yoh_
	X,Y_ = sample_gmm_2d(6, 2, 10)

	clf = KSVMWrap(X, Y_, c=1, gamma='auto')

	# Predicted classes
	Y = clf.predict(X)

	#Print performance (precision and recall for each class)
	accuracy, pr, M = eval_perf_multi(Y, Y_)
	print("Accuracy: ", accuracy)
	print("Precision / Recall")
	for i in range(len(pr)):
		print("Class {} : {}".format(i, pr[i]))
	print("Confusion Matrix:\n ", M)

	#Plot results
	rect=(np.min(X, axis=0), np.max(X, axis=0))
	graph_surface(clf.predict, rect, offset=0)
	graph_data(X, Y_, Y, special=clf.support())
	plt.show()