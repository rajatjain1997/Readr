from functools import partial
import numpy as np

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

inferenceEta = {
	"Z": partial(getLineMembership, (0.0000), (0.00025)),
	"S": partial(getTriangularMembership, (0.0000), (0.00025), (0.0005)),
	"M":  partial(getTriangularMembership, (0.00025), (0.0005), (0.00075)),
	"L": partial(getLineMembership, (0.0005), (0.00075))
}

inferenceAlpha = {
	"Z": partial(getLineMembership, (0.0000), (0.00025)),
	"S": partial(getTriangularMembership, (0.0000), (0.00025), (0.0005)),
	"M":  partial(getTriangularMembership, (0.00025), (0.0005), (0.00075)),
	"L": partial(getLineMembership, (0.0005), (0.00075))
}

inferenceChangeError = {
	"NL": partial(getLineMembership, (-0.0005), (-0.00075)),
	"NM": partial(getTriangularMembership, (-0.00075), (-0.0005), (-0.00025)),
	"NS":  partial(getTriangularMembership, (-0.0005), (-0.00025), (0.0000)),
	"Z": partial(getTriangularMembership, (-0.00025), (0.0000), (0.00025)),
	"PS":  partial(getTriangularMembership, (0.0000), (0.00025), (0.0005)),
	"PM": partial(getTriangularMembership, (0.00025), (0.0005), (0.00075)),
	"PL": partial(getLineMembership, (0.0005), (0.00075))
}

inferenceError = {
	"NL": partial(getLineMembership, (-0.0005), (-0.00075)),
	"NM": partial(getTriangularMembership, (-0.00075), (-0.0005), (-0.00025)),
	"NS":  partial(getTriangularMembership, (-0.0005), (-0.00025), (0.0000)),
	"Z": partial(getTriangularMembership, (-0.00025), (0.0000), (0.00025)),
	"PS":  partial(getTriangularMembership, (0.0000), (0.00025), (0.0005)),
	"PM": partial(getTriangularMembership, (0.00025), (0.0005), (0.00075)),
	"PL": partial(getLineMembership, (0.0005), (0.00075))
}

rulebaseEta = {
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

rulebaseAlpha = {
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
	if (scales == {}):
		return {k: inferenceList[k](x) for k in inferenceList}
	else:
		return {k: scales[k]*inferenceList[k](x) for k in inferenceList}

def ruleAlpha (classE, classCE):
	return rulebaseAlpha[classE][classCE]

def ruleEta (classE, classCE):
	return rulebaseEta[classE][classCE]

def fuzzifyAlpha(e,ce):
	inferDictE = infer(inferenceError,e)
	inferDictCE = infer(inferenceChangeError,ce)
	fuzzifiedMap = {"Z": 0, "S": 0, "M": 0, "L": 0}

	for i in inferDictE:
		for j in inferDictCE:
			fuzzifiedMap[ruleAlpha(i, j)] = max(fuzzifiedMap[ruleAlpha(i,j)] ,min(inferDictE[i],inferDictCE[j]))

	return fuzzifiedMap

def fuzzifyEta(e,ce):
	inferDictE = infer(inferenceError,e)
	inferDictCE = infer(inferenceError,ce)
	fuzzifiedMap = {"Z": 0, "S": 0, "M": 0, "L": 0}

	for i in inferDictE:
		for j in inferDictCE:
			fuzzifiedMap[ruleEta(i, j)] = max(fuzzifiedMap[ruleEta(i,j)] ,min(inferDictE[i],inferDictCE[j]))

	return fuzzifiedMap

def defuzzifyAlpha(scales,interval,start,end):
	finalOutputNum = 0.0
	finalOutputDen = 0.0
	for z in np.arange(start,end,interval):
		inferDict = infer(inferenceAlpha,z,scales)
		finalOutputNum = finalOutputNum+z*max(inferDict.values())
		finalOutputDen = finalOutputDen+max(inferDict.values())
	return (finalOutputNum/finalOutputDen)

def defuzzifyEta(scales,interval,start,end):
	finalOutputNum = 0.0
	finalOutputDen = 0.0
	for z in np.arange(start,end,interval):
		inferDict = infer(inferenceEta,z,scales)
		finalOutputNum = finalOutputNum+z*max(inferDict.values())
		finalOutputDen = finalOutputDen+max(inferDict.values())
	return (finalOutputNum/finalOutputDen)


def alpha(e, ce):
	scales = fuzzifyAlpha(e, ce)
	return defuzzifyAlpha(scales, 0.25,0.0, 1.0)

def eta(e, ce):
	scales = fuzzifyEta(e, ce)
	return defuzzifyEta(scales, 0.25,0.0, 1.0)

# print(eta(-0.25,-0.5))