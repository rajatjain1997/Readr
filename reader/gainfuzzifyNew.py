import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from errorTuning import *

# s1 = ctrl.Antecedent(np.arange(-2.0,3.0,1.0), 's1')
# s2 = ctrl.Antecedent(np.arange(-1.0,2.0,1.0), 's2')
s1 = ctrl.Antecedent(np.arange(-100.0,100.0,0.00001), 's1')
s2 = ctrl.Antecedent(np.arange(-100.0,100.0,0.00001), 's2')
z = ctrl.Consequent(np.arange(0.0,5.0,0.1), 'z')

s1['NB'] = fuzz.trapmf(s1.universe, [-100.0, -100.0, -2.0, -1.0])
s1['NS'] = fuzz.trimf(s1.universe, [-2.0, -1.0, 0.0])
s1['Z'] = fuzz.trimf(s1.universe, [-1.0, 0.0, 1.0])
s1['PS'] = fuzz.trimf(s1.universe, [0.0, 1.0, 2.0])
s1['PB'] = fuzz.trapmf(s1.universe, [1.0, 2.0, 100.0, 100.0])
# s1.view()
# while True:
# 	pass


s2['N'] = fuzz.trapmf(s2.universe, [-100.0, -100.0, -1.0, 0.0])
s2['Z'] = fuzz.trimf(s2.universe, [-1.0, 0.0, 1.0])
s2['P'] = fuzz.trapmf(s2.universe, [0.0, 1.0, 100.0, 100.0])

z['L'] = fuzz.trimf(z.universe, [0.0, 0.0, 2.0])
z['M'] = fuzz.trapmf(z.universe, [0.0, 0.5, 3.5, 4.0])
z['H'] = fuzz.trimf(z.universe, [2.0, 4.0, 4.0])

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
gainFuzzifier = ctrl.ControlSystemSimulation(gainControlSystem)

def gain(S1, S2, convergence):
	scale = errorTuner(S1,S2,convergence)
	print(scale, (10**scale)*S1, (10**scale)*S2)
	gainFuzzifier.input['s1'] = (10**scale)*S1
	gainFuzzifier.input['s2'] = (10**scale)*S2
	gainFuzzifier.compute()
	return gainFuzzifier.output['z']

print(gain(0.0001,0.0001,0.1))