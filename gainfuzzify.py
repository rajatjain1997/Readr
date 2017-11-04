from functools import partial
import numpy as np
import tensorflow as tf

def getTriangularMembership (start, tip, end, x):
	return tf.cond(tf.less(x,start), lambda: tf.constant(0.0), 
								lambda: tf.cond(tf.less_equal(x,tip), lambda: tf.divide(tf.subtract(x,start),tf.subtract(tip,start)), 
															lambda: tf.cond(tf.less_equal(x,end),	lambda: tf.divide(tf.subtract(end,x),tf.subtract(end,tip)),
																						lambda: tf.constant(0.0))))
	# if (x < start):
	# 	return 0
	# elif (x <= tip):
	# 	return (x-start)/(tip-start)
	# elif (x <= end):
	# 	return (end-x)/(end-tip)
	# else:
	# 	return 0

def getLineMembership (base, height, x):
	return tf.cond(tf.greater(height,base), lambda: tf.cond(tf.less(x,base), lambda: tf.constant(0.0), 
																lambda: tf.cond(tf.less(x,height), 	lambda: tf.divide(tf.subtract(x,base),tf.subtract(height,base)),
																								lambda: tf.constant(1.0))), 
									lambda: tf.cond(tf.less(x,height), 	lambda: tf.constant(1.0), 
																	lambda: tf.cond(tf.less(x,base), lambda: tf.divide(tf.subtract(base,x),tf.subtract(base,height)),
																								lambda: tf.constant(0.0))))
	# if (height > base):
	# 	if (x < base):
	# 		return 0
	# 	elif (x < height):
	# 		return (x-base)/(height-base)
	# 	else:
	# 		return 1
	# else:
	# 	if (x < height):
	# 		return 1
	# 	elif (x < base):
	# 		return (base-x)/(base-height)
	# 	else:
	# 		return 0

def getTrapMembership (start, tip1, tip2, end, x):
	return tf.cond(tf.less(x,start), lambda: tf.constant(0.0), 
								lambda: tf.cond(tf.less_equal(x,tip1), 	lambda: tf.divide(tf.subtract(x,start),tf.subtract(tip1,start)), 
																lambda: tf.cond(tf.less_equal(x,tip2),lambda: tf.constant(1.0),
																							tf.cond(tf.less_equal(x,end), lambda: tf.divide(tf.subtract(end,x),tf.subtract(end,tip2)),
																												lambda: tf.constant(0.0)))))
	# if (x < start):
	# 	return 0
	# elif (x <= tip1):
	# 	return (x - start)/(tip1 - start)
	# elif (x <= tip2):
	# 	return 1
	# elif (x <= end):
	# 	return (end - x)/(end - tip2)
	# else:
	# 	return 0

inferenceS1 = {
	"NB": partial(getLineMembership, tf.constant(-1.0), tf.constant(-2.0)),
	"NS": partial(getTriangularMembership, tf.constant(-2.0), tf.constant(-1.0), tf.constant(0.0)),
	"Z":  partial(getTriangularMembership, tf.constant(-1.0), tf.constant(0.0), tf.constant(1.0)),
	"PS": partial(getTriangularMembership, tf.constant(0.0), tf.constant(1.0), tf.constant(2.0)),
	"PB": partial(getLineMembership, tf.constant(1.0), tf.constant(2.0))
}

inferenceS2 = {
	"N": partial(getLineMembership, tf.constant(0.0), tf.constant(-1.0)),
	"Z": partial(getTriangularMembership, tf.constant(-1.0), tf.constant(0.0), tf.constant(1.0)),
	"P": partial(getLineMembership, tf.constant(0.0), tf.constant(1.0))
}

inferenceZ = {
	"L": partial(getLineMembership, tf.constant(2.0), tf.constant(0.0)),
	"M": partial(getTrapMembership, tf.constant(0.0), tf.constant(0.5), tf.constant(3.5), tf.constant(4.0)),
	"H": partial(getLineMembership, tf.constant(2.0), tf.constant(4.0))
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
	inferDictS1 = infer(inferenceS1,s1)
	inferDictS2 = infer(inferenceS2,s2)
	fuzzifiedMap = {"L": tf.Variable(0.0), "M": tf.Variable(0.0), "H": tf.Variable(0.0)}
	S1keys = inferenceS1.keys()
	S2keys = inferDictS2.keys()
	i = tf.constant(1)
	S1Cond = lambda i: i < len(S1keys)
	S2Cond = lambda j: j < len(S2keys)

	def outerBody(i):
		j = tf.constant(1)
		tf.while_loop(S2Cond, innerBody, [j])
		return tf.add(i, 1)

	def innerBody(j):
		tf.assign(fuzzifiedMap[rule(S1Keys[i], S2Keys[j])],tf.maximum(fuzzifiedMap[rule(S1Keys[i],S2Keys[j])] ,tf.minimum(inferDictS1[S1Keys[i]],inferDictS2[S2Keys[j]])))
		return tf.add(j, 1)

	tf.while_loop(S1Cond, outerBody, [i])

	# for i in inferDictS1:
	# 	for j in inferDictS2:
	# 		fuzzifiedMap[rule(i, j)] = tf.maximum(fuzzifiedMap[rule(i,j)] ,tf.minimum(inferDictS1[i],inferDictS2[j])

	return fuzzifiedMap

def defuzzify(scales,interval,start,end):
	finalOutputNum = tf.Variable(0.0)
	finalOutputDen = tf.Variable(0.0)
	for z in np.arange(start,end,interval):
		inferDict = infer(inferenceZ,tf.constant(z),scales)
		finalOutputNum = tf.add(finalOutputNum,tf.multiply(tf.mutiply(tf.constant(z),max(inferDict.values()))))
		finalOutputDen = max(inferDict.values())
	return tf.divide(finalOutputNum,finalOutputDen)

def gain(S1, S2):
	scales = fuzzify(S1, S2)
	return defuzzify(scales, tf.constant(0.1), tf.constant(-2), tf.constant(6))