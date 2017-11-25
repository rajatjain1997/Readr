import reader.reader as rd
import matplotlib.pyplot as plt

def plotComparison2():
	# To plot a comparison between convegence with momentum over mnist data
	session3 = rd.session()
	arr=rd.provideMnistTraining(session3, 1, False,False, False ,0.5)
	plt.plot(arr,label="With Momentum Constant")

	plt.set_yLabel="convergence minimum"
	plt.set_xlabel="Number of iterations"
	plt.legend()
	plt.show()
	plt.savefig("PlotForMnistDataComparison.png")

# print(rd.outputCharacter(session, "./test_pics/8.jpg"))

plotComparison2()