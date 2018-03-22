import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

s1 = ctrl.Antecedent(np.arange(-10.0,11.0,0.1), 's1')
convergence = ctrl.Antecedent(np.arange(0.0,1.1,0.1), 'convergence')
z = ctrl.Consequent(np.arange(-10.0,11.0,1.0), 'z')

s1['L'] = fuzz.trimf(s1.universe, [-10.0, -10.0, 0.0])
s1['M'] = fuzz.trimf(s1.universe, [-10.0, 0.0, 10.0])
s1['H'] = fuzz.trimf(s1.universe, [0.0, 10.0, 10.0])

convergence['L'] = fuzz.trapmf(convergence.universe, [0.0, 0.0, 0.4, 0.5])
convergence['M'] = fuzz.trapmf(convergence.universe, [0.4, 0.5, 0.7, 0.8])
convergence['H'] = fuzz.trapmf(convergence.universe, [0.7, 0.8, 1.0, 1.0])

z['L'] = fuzz.trimf(z.universe, [-10.0, -10.0, -10.0, -10.0])
z['M'] = fuzz.trapmf(z.universe, [0.0, 0.0, 0.0, 0.0])
z['H'] = fuzz.trapmf(z.universe, [10.0, 10.0, 10.0, 10.0])
z.view()
while True:
	pass

rules = []
rules.append(ctrl.Rule(s1['L'] & convergence['L'], z['H']))
rules.append(ctrl.Rule(s1['L'] & convergence['M'], z['H']))
rules.append(ctrl.Rule(s1['L'] & convergence['H'], z['M']))
rules.append(ctrl.Rule(s1['M'] & convergence['L'], z['M']))
rules.append(ctrl.Rule(s1['M'] & convergence['M'], z['M']))
rules.append(ctrl.Rule(s1['M'] & convergence['H'], z['M']))
rules.append(ctrl.Rule(s1['H'] & convergence['L'], z['M']))
rules.append(ctrl.Rule(s1['H'] & convergence['M'], z['L']))
rules.append(ctrl.Rule(s1['H'] & convergence['H'], z['L']))

gainControlSystem = ctrl.ControlSystem(rules)
gainFuzzifier = ctrl.ControlSystemSimulation(gainControlSystem)

def getPower(number):
	power = 0;
	number = abs(number)
	if number < 1:
		while not(number >= 1):
			number = number*10
			power-=1
	else:  
		while not(number < 10):
			number = number/10
			power+=1

	return power

def errorTuner(S1, S2, convergence):
	if abs(getPower(S1)) > abs(getPower(S2)):
		gainFuzzifier.input['s1'] = getPower(S1)
		print(getPower(S1))
	else:
		gainFuzzifier.input['s1'] = getPower(S2)
		print(getPower(S2))
	
	gainFuzzifier.input['convergence'] = convergence
	gainFuzzifier.compute()
	# z.view(sim=gainFuzzifier)
	gainFuzzifier.print_state()
	return gainFuzzifier.output['z']

# print(errorTuner(0.01,0.01,0.0))