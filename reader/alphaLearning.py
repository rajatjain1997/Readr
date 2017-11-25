from functools import partial
import numpy as np

## this module includes functions for the fuzzification and defuzzification of the momentum constant and learning rate

def getTriangularMembership (start, tip, end, x):
	#computes membership of a point given the parameters of the triangular membership function 
	if (x < start):
		return 0
	elif (x <= tip):
		return (x-start)/(tip-start)
	elif (x <= end):
		return (end-x)/(end-tip)
	else:
		return 0

def getLineMembership (base, height, x):
	#computes membership of a point given the parameters of the line membership function 
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
	#computes membership of a point given the parameters of the trapezium membership function 
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


# Gives the membership values of a point in Eta(learning rate) fuzzy set
inferenceEta = {
	"Z": partial(getLineMembership, (0.0000), (0.00025)),
	"S": partial(getTriangularMembership, (0.0000), (0.00025), (0.0005)),
	"M":  partial(getTriangularMembership, (0.00025), (0.0005), (0.00075)),
	"L": partial(getLineMembership, (0.0005), (0.00075))
}

# Gives the membership values of a point in Alpha(momentum constant) fuzzy set
inferenceAlpha = {
	"Z": partial(getLineMembership, (0.0000), (0.00025)),
	"S": partial(getTriangularMembership, (0.0000), (0.00025), (0.0005)),
	"M":  partial(getTriangularMembership, (0.00025), (0.0005), (0.00075)),
	"L": partial(getLineMembership, (0.0005), (0.00075))
}

# Gives the membership values of a point for the change in error fuzzy set
inferenceChangeError = {
	"NL": partial(getLineMembership, (-0.0005), (-0.00075)),
	"NM": partial(getTriangularMembership, (-0.00075), (-0.0005), (-0.00025)),
	"NS":  partial(getTriangularMembership, (-0.0005), (-0.00025), (0.0000)),
	"Z": partial(getTriangularMembership, (-0.00025), (0.0000), (0.00025)),
	"PS":  partial(getTriangularMembership, (0.0000), (0.00025), (0.0005)),
	"PM": partial(getTriangularMembership, (0.00025), (0.0005), (0.00075)),
	"PL": partial(getLineMembership, (0.0005), (0.00075))
}

# Gives the membership values of a point for the error fuzzy set
inferenceError = {
	"NL": partial(getLineMembership, (-0.0005), (-0.00075)),
	"NM": partial(getTriangularMembership, (-0.00075), (-0.0005), (-0.00025)),
	"NS":  partial(getTriangularMembership, (-0.0005), (-0.00025), (0.0000)),
	"Z": partial(getTriangularMembership, (-0.00025), (0.0000), (0.00025)),
	"PS":  partial(getTriangularMembership, (0.0000), (0.00025), (0.0005)),
	"PM": partial(getTriangularMembership, (0.00025), (0.0005), (0.00075)),
	"PL": partial(getLineMembership, (0.0005), (0.00075))
}

# generic rulebase
rulebaseEta = {
	# key : error 
	# value : a map for which the key is change in error and the value is the corresponding Eta
	"NL": {
		"PL": "L",
		"PM": "L",
		"PS": "L",
		"Z":"L",
		"NS":"L",
		"NM":"L",
		"NL":"L"
	},
	"NM": {
		"PL": "M",
		"PM": "M",
		"PS": "M",
		"Z":"M",
		"NS":"L",
		"NM":"L",
		"NL":"L"
	},
	"NS": {
		"PL": "S",
		"PM": "S",
		"PS": "S",
		"Z":"S",
		"NS":"L",
		"NM":"L",
		"NL":"L"
	},
	"Z": {
		"PL": "Z",
		"PM": "Z",
		"PS": "Z",
		"Z":"Z",
		"NS":"Z",
		"NM":"Z",
		"NL":"Z"
	},
	"PS": {
		"PL": "L",
		"PM": "L",
		"PS": "L",
		"Z":"S",
		"NS":"S",
		"NM":"S",
		"NL":"S"
	},
	"PM": {
		"PL": "L",
		"PM": "L",
		"PS": "L",
		"Z":"M",
		"NS":"M",
		"NM":"M",
		"NL":"M"
	},
	"PL": {
		"PL": "L",
		"PM": "L",
		"PS": "L",
		"Z":"L",
		"NS":"L",
		"NM":"L",
		"NL":"L"
	}
}

# generic rulebase
rulebaseAlpha = {
	# key : error 
	# value : a map for which the key is change in error and the value is the corresponding Alpha
	"NL": {
		"PL": "L",
		"PM": "L",
		"PS": "L",
		"Z":"L",
		"NS":"Z",
		"NM":"Z",
		"NL":"Z"
	},
	"NM": {
		"PL": "M",
		"PM": "M",
		"PS": "M",
		"Z":"M",
		"NS":"Z",
		"NM":"Z",
		"NL":"Z"
	},
	"NS": {
		"PL": "S",
		"PM": "S",
		"PS": "S",
		"Z":"S",
		"NS":"Z",
		"NM":"Z",
		"NL":"Z"
	},
	"Z": {
		"PL": "Z",
		"PM": "Z",
		"PS": "Z",
		"Z":"Z",
		"NS":"Z",
		"NM":"Z",
		"NL":"Z"
	},
	"PS": {
		"PL": "Z",
		"PM": "Z",
		"PS": "Z",
		"Z":"S",
		"NS":"S",
		"NM":"S",
		"NL":"S"
	},
	"PM": {
		"PL": "Z",
		"PM": "Z",
		"PS": "Z",
		"Z":"M",
		"NS":"M",
		"NM":"M",
		"NL":"M"
	},
	"PL": {
		"PL": "Z",
		"PM": "Z",
		"PS": "Z",
		"Z":"L",
		"NS":"L",
		"NM":"L",
		"NL":"L"
	}
}

def infer (inferenceList, x, scales = {}):
	# if scales = {} then returns the membership values of the input point x from the inference map provided in inferenceList
	# if scales != {} then returns the (membership values*corresponding value in scales) of the input point x from the inference map provided in inferenceList
	if (scales == {}):
		return {k: inferenceList[k](x) for k in inferenceList}
	else:
		return {k: scales[k]*inferenceList[k](x) for k in inferenceList}

def ruleAlpha (classE, classCE):
	# returns the corresponding Alpha fired from given error and change in error
	return rulebaseAlpha[classE][classCE]

def ruleEta (classE, classCE):
	# returns the corresponding Eta fired from given error and change in error
	return rulebaseEta[classE][classCE]

def fuzzifyAlpha(e,ce):
	# returns a map containing fuzzified outputs for Alpha with corresponding membership values given error and change in error as input
	inferDictE = infer(inferenceError,e)
	inferDictCE = infer(inferenceChangeError,ce)
	fuzzifiedMap = {"Z": 0, "S": 0, "M": 0, "L": 0}

	for i in inferDictE:
		for j in inferDictCE:
			fuzzifiedMap[ruleAlpha(i, j)] = max(fuzzifiedMap[ruleAlpha(i,j)] ,min(inferDictE[i],inferDictCE[j]))

	return fuzzifiedMap

def fuzzifyEta(e,ce):
	# returns a map containing fuzzified outputs for Eta with corresponding membership values given error and change in error as input
	inferDictE = infer(inferenceError,e)
	inferDictCE = infer(inferenceError,ce)
	fuzzifiedMap = {"Z": 0, "S": 0, "M": 0, "L": 0}

	for i in inferDictE:
		for j in inferDictCE:
			fuzzifiedMap[ruleEta(i, j)] = max(fuzzifiedMap[ruleEta(i,j)] ,min(inferDictE[i],inferDictCE[j]))

	return fuzzifiedMap

def defuzzifyAlpha(scales,interval,start,end):
	# gives the defuzzified output for Alpha using centoid defuzzification given fuzzified outputs as scales
	finalOutputNum = 0.0
	finalOutputDen = 0.0
	for z in np.arange(start,end,interval):
		inferDict = infer(inferenceAlpha,z,scales)
		finalOutputNum = finalOutputNum+z*max(inferDict.values())
		finalOutputDen = finalOutputDen+max(inferDict.values())
	return (finalOutputNum/finalOutputDen)

def defuzzifyEta(scales,interval,start,end):
	# gives the defuzzified output for Eta using centoid defuzzification given fuzzified outputs as scales
	finalOutputNum = 0.0
	finalOutputDen = 0.0
	for z in np.arange(start,end,interval):
		inferDict = infer(inferenceEta,z,scales)
		finalOutputNum = finalOutputNum+z*max(inferDict.values())
		finalOutputDen = finalOutputDen+max(inferDict.values())
	return (finalOutputNum/finalOutputDen)


def alpha(e, ce):
	# integrates the entire fuzzification-defuzzification process for momentum constant
	scales = fuzzifyAlpha(e, ce)
	return defuzzifyAlpha(scales, 0.25,0.0, 1.0)

def eta(e, ce):
	# integrates the entire fuzzification-defuzzification process for learning rate
	scales = fuzzifyEta(e, ce)
	return defuzzifyEta(scales, 0.25,0.0, 1.0)