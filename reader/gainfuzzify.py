from functools import partial
import numpy as np
import tensorflow as tf

def getTriangularMembership (start, tip, end, x):
	if (x < start):
		return 0
	elif (x <= tip):
		return (x-start)/(tip-start)
	elif (x <= end):
		return (end-x)/(end-tip)
	else:
		return 0

def getLineMembership (base, height, x):
	if (height > base):
		if (x < base):
			return 0
		elif (x < height):
			return (x-base)/(height-base)
		else:
			return 1
	else:
		if (x < height):
			return 1
		elif (x < base):
			return (base-x)/(base-height)
		else:
			return 0

def getTrapMembership (start, tip1, tip2, end, x):
	if (x < start):
		return 0
	elif (x <= tip1):
		return (x - start)/(tip1 - start)
	elif (x <= tip2):
		return 1
	elif (x <= end):
		return (end - x)/(end - tip2)
	else:
		return 0
def changeSensitivity():
	global sensitivity
	# print("s=",sensitivity)
	global inferenceS1
	global inferenceS2
	global inferenceZ
	inferenceS1 = {
		"NB": partial(getLineMembership, (-1.0/sensitivity), (-2.0/sensitivity)),
		"NS": partial(getTriangularMembership, (-2.0/sensitivity), (-1.0/sensitivity), (0.0/sensitivity)),
		"Z":  partial(getTriangularMembership, (-1.0/sensitivity), (0/sensitivity), (1.0/sensitivity)),
		"PS": partial(getTriangularMembership, (0.0/sensitivity), (1.0/sensitivity), (2.0/sensitivity)),
		"PB": partial(getLineMembership, (1.0/sensitivity), (2.0/sensitivity))
	}

	inferenceS2 = {
		"N": partial(getLineMembership, (0.0/sensitivity), (-1.0/sensitivity)),
		"Z": partial(getTriangularMembership, (-1.0/sensitivity), (0.0/sensitivity), (1.0/sensitivity)),
		"P": partial(getLineMembership, (0.0/sensitivity), (1.0/sensitivity))
	}

	inferenceZ = {
		"L": partial(getLineMembership, (2.0), (0.0)),
		"M": partial(getTrapMembership, (0.0), (0.5), (3.5), (4.0)),
		"H": partial(getLineMembership, (2.0), (4.0))
	}

rulebase = {
	"NB": {
		"N": "L",
		"Z": "L",
		"P": "L"
	},
	"NS": {
		"N": "L",
		"Z": "M",
		"P": "M"
	},
	"Z": {
		"N": "M",
		"Z": "M",
		"P": "M"
	},
	"PS": {
		"N": "M",
		"Z": "M",
		"P": "H"
	},
	"PB": {
		"N": "H",
		"Z": "H",
		"P": "H"
	}
}

def infer (inferenceList, x, scales = {}):
	if (scales == {}):
		return {k: inferenceList[k](x) for k in inferenceList}
	else:
		return {k: scales[k]*inferenceList[k](x) for k in inferenceList}

def rule (classS1, classS2):
	return rulebase[classS1][classS2]

def fuzzify(s1,s2):
	global inferenceS1
	global inferenceS2
	global inferenceZ
	inferDictS1 = infer(inferenceS1,s1)
	inferDictS2 = infer(inferenceS2,s2)
	fuzzifiedMap = {"L": 0, "M": 0, "H": 0}

	for i in inferDictS1:
		for j in inferDictS2:
			fuzzifiedMap[rule(i, j)] = max(fuzzifiedMap[rule(i,j)] ,min(inferDictS1[i],inferDictS2[j]))

	return fuzzifiedMap

def defuzzify(scales,interval,start,end):
	finalOutputNum = 0.0
	finalOutputDen = 0.0
	for z in np.arange(start,end,interval):
		inferDict = infer(inferenceZ,z,scales)
		finalOutputNum = finalOutputNum+z*max(inferDict.values())
		finalOutputDen = finalOutputDen+max(inferDict.values())
	return (finalOutputNum/finalOutputDen)

def gain(S1, S2,sens):
	global sensitivity
	sensitivity=sens;
	changeSensitivity()
	scales = fuzzify(S1, S2)
	return defuzzify(scales, 1,-2, 6)