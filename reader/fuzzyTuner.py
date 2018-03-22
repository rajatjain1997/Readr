import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

R = ctrl.Antecedent(np.arange(0.0,1.0,0.1), 'R')
z = ctrl.Consequent(np.arange(-1.0,1.0,0.33), 'z')

R['ZE'] = fuzz.trimf(R.universe, [0.0, 0.0, 0.1])
R['VS'] = fuzz.trimf(R.universe, [0.0, 0.1, 0.25])
R['S'] = fuzz.trimf(R.universe, [0.1, 0.25, 0.49])
R['M-'] = fuzz.trimf(R.universe, [0.25, 0.49, 0.51])
R['M+'] = fuzz.trimf(R.universe, [0.49, 0.51, 0.75])
R['B'] = fuzz.trimf(R.universe, [0.51, 0.75, 1.0])
R['VB'] = fuzz.trimf(R.universe, [0.75, 1.0, 1.0])


z['NB'] = fuzz.trimf(z.universe, [-1.0, -1.0, -1.0])
z['NM'] = fuzz.trimf(z.universe, [-0.66, -0.66, -0.66])
z['NS'] = fuzz.trimf(z.universe, [-0.33, -0.33, -0.33])
z['ZE'] = fuzz.trimf(z.universe, [0.0, 0.0, 0.0])
z['PS'] = fuzz.trimf(z.universe, [0.33, 0.33, 0.33])
z['PM'] = fuzz.trimf(z.universe, [0.66, 0.66, 0.66])
z['PB'] = fuzz.trimf(z.universe, [1.0, 1.0, 1.0])

rules = []
rules.append(ctrl.Rule(R['ZE'], z['ZE']))
rules.append(ctrl.Rule(R['VS'], z['ZE']))
rules.append(ctrl.Rule(R['S'], z['NS']))
rules.append(ctrl.Rule(R['M-'], z['NM']))
rules.append(ctrl.Rule(R['M+'], z['ZE']))
rules.append(ctrl.Rule(R['B'], z['PM']))
rules.append(ctrl.Rule(R['VB'], z['PB']))
gainControlSystem = ctrl.ControlSystem(rules)
gainFuzzifier = ctrl.ControlSystemSimulation(gainControlSystem)

def tuner(convergence, setPoint):
	gainFuzzifier.input['R'] = abs(setPoint - convergence)/setPoint
	gainFuzzifier.compute()
	T = gainFuzzifier.output['z']


# print(gain(1.0,1.0))