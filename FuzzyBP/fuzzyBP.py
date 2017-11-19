import itertools
from math import e

def vectorMul(a, b):
	assert(len(a)==len(b))
	ret = 0
	for i in range(len(a)):
		ret += a[i]*b[i]
	return ret

def fnet(net):
	return 1.0 / (1.0 + e**(-4.0*(net-0.5)))

def FuzzyBP(inputs, weights, target):
	hidden_inputs = []
	for i in range(len(weights[0][0])):
		val = 0
		for m in range(1,len(inputs)+1):
			if m == len(inputs):
				val = max(min(inputs), val)
				break
			for s in itertools.combinations(range(len(inputs)), m):
				nval = max([weights[0][x][i] for x in s]) # g(G)
				val = max(min([inputs[x] for x in s]+[nval]), val) # minimum of g(G) and x in G
		hidden_inputs.append(fnet(val))

	foutput = []
	for i in range(len(weights[1][0])):
		output = 0
		for m in range(1,len(hidden_inputs)+1):
			if m == len(hidden_inputs):
				output = max(min(hidden_inputs), output)
				break
			for s in itertools.combinations(range(len(hidden_inputs)), m):
				nval = max([weights[1][x][i] for x in s]) # g(G)
				output = max(min([hidden_inputs[x] for x in s]+[nval]), output) # minimum of g(G) and x in G
		foutput.append(fnet(output))

	learning_rate = 0.1
	output = np.argmax(foutput)+1
	if target != output:
		for i in range(len(weights[0])):
			for j in range(len(weights[0][i])):
				if weights[0][i][j] < target and target > output:
					weights[0][i][j] = min(1, weights[0][i][j] + learning_rate * abs(hidden_inputs[j]-target))
				if weights[0][i][j] > target and target < output:
					weights[0][i][j] = max(0, weights[0][i][j] - learning_rate * abs(hidden_inputs[j]-target))
		for i in range(len(weights[1])):
			for j in range(len(weights[1][i])):
				if weights[1][0][j] < target and target > output:
					weights[1][0][j] = min(1, weights[1][0][j] + learning_rate * abs(output-target))
				if weights[1][0][j] > target and target < output:
					weights[1][0][j] = max(0, weights[1][0][j] - learning_rate * abs(output-target))
	return weights

#print(FuzzyBP([1,0,0,1],[[[1,4],[5,3],[3,7],[2,53]],[[31],[12]]],8))