import reader.reader as rd
import matplotlib.pyplot as plt

def plotComparison():
	# To plot a comparison between convergence with and without gain Fuzzification
	session = rd.session()
	arr=rd.train(session, "./test_pics/8.jpg", 8)
	plt.plot(arr,label="With gain fuzzification")

	session2 = rd.session()
	arr2=rd.train(session2, "./test_pics/8.jpg", 8,False)
	plt.plot(arr2,label="Without gain Fuzzification")

	plt.set_yLabel="convergence minimum"
	plt.set_xlabel="Number of iterations"
	plt.legend()
	plt.show()
	plt.savefig("PlotFor8.png")

def plotComparison3():
	# To plot convergence with eta and momentum Fuzzification
	session = rd.session()
	arr=rd.train2(session, "./test_pics/8.jpg", 8,False,True,True)
	plt.plot(arr,label="With Momentum and ETA fuzzification")

	plt.set_yLabel="convergence minimum"
	plt.set_xlabel="Number of iterations"
	plt.legend()
	plt.show()
	plt.savefig("PlotFor8.png")

plotComparison()