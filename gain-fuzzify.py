from functools import partial

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

inferenceS1 = {
	"NB": partial(getLineMembership, -1, -2),
	"NS": partial(getTriangularMembership, -2, -1, 0),
	"Z": partial(getTriangularMembership, -1, 0, 1),
	"PS": partial(getTriangularMembership, 0, 1, 2),
	"PB": partial(getLineMembership, 1, 2)
}

inferenceS2 = {
	"N": partial(getLineMembership, 0, -1),
	"Z": partial(getTriangularMembership, -1, 0, 1),
	"P": partial(getLineMembership, 0, 1)
}

inferenceZ = {
	"L": partial(getLineMembership, 2, 0),
	"M": partial(getTrapMembership, 0, 0.5, 3.5, 4),
	"H": partial(getLineMembership, 2, 4)
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
	return rulebase[S1][S2]

def fuzzify(s1,s2):
	inferDictS1 = infer(inferenceS1,s1)
	inferDictS2 = infer(inferenceS2,s2)
	fuzzifiedMap = {}
	
	for i in inferDictS1:
		for j in inferDictS2:
			fuzzifiedMap[rule(i,j)] = min(inferenceS1[i],inferenceS2[j])

	return fuzzifiedMap

def defuzzify(L,M,H,interval,start,end):
	finalOutputNum = 0.0
	finalOutputDen = 0.0
	
	for z in range(start,end,interval):
		inferDict = infer(inferenceZ,z,L,M,H)
		finalOutputNum += z*max(inferDict.values())
		finalOutputDen += max(inferDict.values())
  	
  	return (finalOutputNum/finalOutputDen)

