import reader.reader as rd
import matplotlib.pyplot as plt

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
# print(rd.outputCharacter(session, "./test_pics/8.jpg"))