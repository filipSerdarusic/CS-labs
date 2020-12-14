import sys
import re

# States
states = sys.stdin.readline().rstrip().split(',')

# Alphabet
alphabet = sys.stdin.readline().rstrip().split(',')

# Accepted states
accepted_states = sys.stdin.readline().rstrip().split(',')

# Initial state
init_state = sys.stdin.readline().rstrip()

# Transitions
transitions = {}
for line in sys.stdin:
	current_state, symbol, next_state = re.split('\W+', line.rstrip())
	if current_state in transitions.keys():
		transitions[current_state].update({symbol : next_state})
	else:
		transitions[current_state] = {symbol : next_state}


# Minimization

# Finding unreachable states
reachable = [init_state]
stop = False
while (not stop):
	new_reachable = reachable.copy()

	for q in reachable:
		for sym in alphabet:
			p = transitions[q][sym]
			if p not in reachable:
				reachable.append(p)

	if new_reachable == reachable:
		stop = True

# Removing unreachble states
states = [p for p in states if p in reachable]
accepted_states = [p for p in accepted_states if p in reachable]


# Finding distinguishable states
def distinguishable(p,q):
	x = p in accepted_states
	y =  q in accepted_states
	return (x and y) or (not x and not y)

table = {}
for p in states:
	for q in states:
		if p != q:
			table[p,q] = distinguishable(p,q)

change = True
while change:
	change = False
	for key in table.keys():
		if table[key] is True:
			for sym in alphabet:
				p_ = transitions[key[0]][sym]
				q_ = transitions[key[1]][sym]
				if  (p_ != q_) and (table[p_,q_] is False):
					table[key] = False
					change = True

# Removing indistinguishable states
indistinguishable = {}
for (p,q) in table.keys():
	if table[(p,q)] is True:
		if p > q:
			indistinguishable[p] = q
			if p in states:
				states.remove(p)
			if p in accepted_states:
				accepted_states.remove(p)
		else:
			indistinguishable[q] = p
			if q in states:
				states.remove(q)
			if q in accepted_states:
				accepted_states.remove(q)

if init_state not in states:
	init_state = indistinguishable[init_state]

print(','.join(states))
print(','.join(alphabet))
print(','.join(accepted_states))
print(init_state)
for p in states:
	for sym in alphabet:
		q = transitions[p][sym]
		while q not in states:
			q = indistinguishable[q]
		print("{},{}->{}".format(p,sym,q))