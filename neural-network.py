import reader.reader as rd
import matplotlib.pyplot as plt
def plotComparison():
	# To plot a comparison between convergence with and without gain Fuzzification
	session = rd.session()
	arr=rd.train(session, "./test_pics/8.jpg", 8)
	plt.plot(arr,label="With Gain Fuzzification")

	session2 = rd.session()
	arr2=rd.train(session2, "./test_pics/8.jpg", 8,False)
	plt.plot(arr2,label="Without Gain Fuzzification")

	plt.set_yLabel="convergence minimum"
	plt.set_xlabel="Number of iterations"
	plt.legend()
	plt.show()
	plt.savefig("PlotFor8.png")

def plotComparison2():
	# To plot a comparison between convegence with gain fuzzification and without and with momentum over mnist data
	session = rd.session()
	arr=rd.provideMnistTraining(session, 10)
	plt.plot(arr,label="With Gain Fuzzification")
	# There is something wrong with this mnist training as convergence is some 80.something % and in the following iterations it drops down and settles at 80%
	# did not check if this is true for the ramining two
	
	session2 = rd.session()
	arr=rd.provideMnistTraining(session2, 10, False)
	plt.plot(arr2,label="Without Gain Fuzzification")


	session3 = rd.session()
	arr=rd.provideMnistTraining(session3, 10, False,0.5)
	plt.plot(arr2,label="With Momentum Constant")

	plt.set_yLabel="convergence minimum"
	plt.set_xlabel="Number of iterations"
	plt.legend()
	plt.show()
	plt.savefig("PlotForMnistDataComparison.png")

# print(rd.outputCharacter(session, "./test_pics/8.jpg"))

plotComparison2()