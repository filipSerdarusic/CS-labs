import sys

i = 1
def next_symbol():
	global symbol, i
	symbol = string[i]
	i += 1

def S():
	global symbol
	print('S', end='')
	if symbol is 'a':
		next_symbol()
		if A():
			if B():
				return True
	elif symbol is 'b':
		next_symbol()
		if B():
			if A():
				return True
	return False

def A():
	global symbol
	print('A', end='')
	if symbol is 'b':
		next_symbol()		
		if C():
			return True
	elif symbol is 'a':
		next_symbol()
		return True
	return False

def B():
	global symbol
	print('B', end='')
	if symbol is 'c':
		next_symbol()
		if symbol is 'c':
			next_symbol()
			if S():
				if symbol is 'b':
					next_symbol()
					if symbol is 'c':
						next_symbol()
						return True
		return False
	return True

def C():
	global symbol
	print('C', end='')
	if A():
		if A():
			return True
	return False

string = list(sys.stdin.read().rstrip())
string.append('')
symbol = string[0]
accept = S() and symbol is ''
print('\nDA' if accept else '\nNE')