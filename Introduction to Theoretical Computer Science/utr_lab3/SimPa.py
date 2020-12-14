import sys
import re

# Input strings
strings = sys.stdin.readline().rstrip().split('|')

# States
states = sys.stdin.readline().rstrip().split(',')

# Alphabet symbols
alphabet = sys.stdin.readline().rstrip().split(',')

# Stack symbols
stack_sym = sys.stdin.readline().rstrip().split(',')

# Accepted states
accepted_states = sys.stdin.readline().rstrip().split(',')

# Initial state
init_state = sys.stdin.readline().rstrip()

# Initial stack symbol
init_stack = sys.stdin.readline().rstrip()

# Transitions
transitions = {}
for line in sys.stdin:
	current_state, sym, stack_sym, next_state, stack_strings = re.split('[^A-Za-z0-9$]+', line.rstrip())
	
	if current_state in transitions.keys():
		transitions[current_state].update({(sym,stack_sym) : (next_state,stack_strings)})
	else:
		transitions[current_state] = {(sym,stack_sym) : (next_state,stack_strings)}

def has_transition(state, sym, stack):
	try:
		if len(stack) == 0:
			stack = '$'
		_ = transitions[state][(sym, stack[0])]
		return True
	except KeyError:
		return False

for input in strings:
	input = input.split(',')
	state = init_state
	stack = init_stack

	output = []
	output.append(state + '#' + stack)

	failed = False
	for sym in input:

		if has_transition(state, sym, stack):
			state, next_stack = transitions[state][(sym, stack[0])]
		else:

			if (len(input) > 1):
				output.append('fail')
				failed = True
				break
			
			else:
				while(True):
					try:
						state, next_stack = transitions[state][('$', stack[0])]
						stack = (next_stack != '$')*next_stack + stack[1:] + (len(stack)==0)*'$'
						output.append(state + 
									'#' + stack + (len(stack)==0)*'$')
					except KeyError:
						break
	
				break

		stack = (next_stack != '$')*next_stack + stack[1:]
		stack = stack + (len(stack)==0)*'$'

		output.append(state + '#' + stack)
		while (next_stack == '$'):
			try:
				state, next_stack = transitions[state][('$', stack[0])]
			except KeyError:
				break
			stack = (next_stack != '$')*next_stack + stack[1:]
			output.append(state + '#' + stack + (len(stack)==0)*'$')
			if len(stack) == 0:
				break


	if failed:
		output.append('0')
	else:
		output.append(str(int(state in accepted_states)))
	print('|'.join(output))