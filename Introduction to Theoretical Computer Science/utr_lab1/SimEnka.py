import sys

# Defining automata
input_strings = []
transitions = {}

# Input strings
line = sys.stdin.readline().rstrip().split('|')
for strings in line:
	if ',' in strings:
		input_strings.append(strings.split(','))
	else:
		input_strings.append(strings)

# States
states = sys.stdin.readline().rstrip().split(',')

# Alphabet
alphabet = sys.stdin.readline().rstrip().split(',')

# Accepted states
accepted_states = sys.stdin.readline().rstrip().split(',')

# Initial states
init_states = sys.stdin.readline().rstrip().split(',')

# Transitions
for line in sys.stdin:
	line = line.rstrip().split('->')
	next_state = line[1]
	line = line[0].split(',')
	current_state = line[0]
	symbol =line[1]

	if ',' in next_state:
		next_state = next_state.split(',')

	if current_state in transitions.keys():
		new_transitions = {}

		for (key, value) in transitions[current_state].items():
			new_transitions[key] = value
		new_transitions[symbol] = next_state
		transitions[current_state] = new_transitions

	else:
		transitions[current_state] = {symbol : next_state}

def has_transition(state, symbol):
	if state in transitions.keys():
		if symbol in transitions[state]:
			return True
		return False

def change_state(state, symbol):
	new_states = transitions[state][symbol]
	return new_states

def eps_transition(states):
	eps_states = states

	no_changes = False
	while(not no_changes):

		for state in states:

			if has_transition(state, '$'):
				new_state = transitions[state]['$']

				if isinstance(new_state, list):
						eps_states.extend(new_state)
						eps_states = list(set(eps_states))
	
				else:
					if new_state not in eps_states and new_state != '#':
						eps_states.append(new_state)

		eps_states = list(set(eps_states))
		
		no_changes = sorted(eps_states) == sorted(states)
		states = sorted(eps_states)

	return states

# Run
for string in input_strings:
	current_states = init_states

	output = []
	current_states = eps_transition(current_states)
	output.append(','.join(sorted(current_states)))

	for symbol in string:
		next_states = []

		#no_transitions = True
		for state in current_states:

			if has_transition(state, symbol):
				new = change_state(state, symbol)
				#no_transitions = False

				if isinstance(new, str):
					if new not in next_states and new != '#':
						next_states.append(new)

				if isinstance(new, list):
					next_states.extend(new)
				
				next_states = list(set(next_states))

		current_states = next_states
		current_states = eps_transition(current_states)

		if len(current_states) == 0:
			new = '#'
			current_states.append(new)

		output.append(','.join(sorted(current_states)))

	output = '|'.join(output)
	print(output)