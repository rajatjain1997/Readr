import functools

def getTriangularMembership (start, tip, end, x):
	if (x < start):
		return 0
	elif (x <= tip):
		return (x-start)/(tip-start)
	elif (x > tip):
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