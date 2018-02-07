import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

def initializeMembershipFunctions():
	s1 = ctrl.Antecedent(np.arange(-2.0,3.0,1.0), 's1')
	s2 = ctrl.Antecedent(np.arange(-1.0,2.0,1.0), 's2')
	z = ctrl.Consequent(np.arange(0.0,5.0,1.0), 'z')
	
	s1['NB'] = fuzz.trimf(s1.universe, [-2.0, -2.0, -1.0])
	s1['NS'] = fuzz.trimf(s1.universe, [-2.0, -1.0, 0.0])
	s1['Z'] = fuzz.trimf(s1.universe, [-1.0, 0.0, 1.0])
	s1['PS'] = fuzz.trimf(s1.universe, [0.0, 1.0, 2.0])
	s1['PB'] = fuzz.trimf(s1.universe, [1.0, 1.0, 2.0])

	s2['N'] = fuzz.trimf(s2.universe, [-1.0, -1.0, 0.0])
	s2['Z'] = fuzz.trimf(s2.universe, [-1.0, 0.0, 1.0])
	s2['P'] = fuzz.trimf(s2.universe, [0.0, 1.0, 1.0])

	z['L'] = fuzz.trimf(z.universe, [0.0, 0.0, 2.0])
	z['M'] = fuzz.trapmf(z.universe, [0.0, 0.5, 3.5, 4.0])
	z['H'] = fuzz.trimf(z.universe, [2.0, 4.0, 4.0])

	# s1.view()
	# s2.view()
	# z.view()

	# while(True):
	# 	pass

	return s1,s2,z

def initializeRules(s1,s2,z):
	rules = []
	rules.append(ctrl.Rule(s1['NB'] & s2['N'], z['L']))
	rules.append(ctrl.Rule(s1['NB'] & s2['Z'], z['L']))
	rules.append(ctrl.Rule(s1['NB'] & s2['P'], z['L']))
	rules.append(ctrl.Rule(s1['NS'] & s2['N'], z['L']))
	rules.append(ctrl.Rule(s1['NS'] & s2['Z'], z['M']))
	rules.append(ctrl.Rule(s1['NS'] & s2['P'], z['M']))
	rules.append(ctrl.Rule(s1['Z'] & s2['N'], z['M']))
	rules.append(ctrl.Rule(s1['Z'] & s2['Z'], z['M']))
	rules.append(ctrl.Rule(s1['Z'] & s2['P'], z['M']))
	rules.append(ctrl.Rule(s1['PS'] & s2['N'], z['M']))
	rules.append(ctrl.Rule(s1['PS'] & s2['Z'], z['M']))
	rules.append(ctrl.Rule(s1['PS'] & s2['P'], z['H']))
	rules.append(ctrl.Rule(s1['PB'] & s2['N'], z['H']))
	rules.append(ctrl.Rule(s1['PB'] & s2['Z'], z['H']))
	rules.append(ctrl.Rule(s1['PB'] & s2['P'], z['H']))
	gainControlSystem = ctrl.ControlSystem(rules)
	return gainControlSystem


def gain(S1, S2):
	s1,s2,z = initializeMembershipFunctions()
	gainControlSystem = initializeRules(s1,s2,z)
	gainFuzzifier = ctrl.ControlSystemSimulation(gainControlSystem)
	gainFuzzifier.input['s1'] = S1
	gainFuzzifier.input['s2'] = S2
	gainFuzzifier.compute()
	return gainFuzzifier.output['z']

# print(gain(1.0,1.0))