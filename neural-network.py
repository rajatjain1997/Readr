import reader.reader as rd
import matplotlib.pyplot as plt
import numpy as np

def plotComparison():
	# To plot a comparison between convergence with and without gain Fuzzification
	session = rd.session()
	arr=rd.train(session, np.fromfile("./data/zero", sep = " "), [0,0,0,1])
	plt.plot(arr,label="With gain fuzzification")

	session2 = rd.session()
	arr2=rd.train(session2, np.fromfile("./data/zero", sep = " "), [0,0,0,1], False)
	plt.plot(arr2,label="Without gain Fuzzification")

	plt.set_yLabel="convergence minimum"
	plt.set_xlabel="Number of iterations"
	plt.legend()
	plt.show()
	plt.savefig("PlotFor8.png")

plotComparison()