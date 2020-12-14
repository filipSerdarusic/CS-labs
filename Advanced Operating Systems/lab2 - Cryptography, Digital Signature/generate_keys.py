from Crypto.PublicKey import RSA

'''
Generating public and private keys for sender and reciever.
Each key is stored in its own file (for example: 'public_key_A.pem')
'''

RSA_KEY_SIZE = 2048		# Key size: 1024, 2048 or 4096

def generate_keys(name, bits=2048):

	key = RSA.generate(bits)
	public_key = key.publickey()

	public_key_file = 'public_key_' + str(name) + '.pem'
	private_key_file = 'private_key_' + str(name) + '.pem'

	with open(public_key_file, 'w') as f:
		f.write(public_key.exportKey('PEM').decode('utf-8'))

	with open(private_key_file, 'w') as f:
		f.write(key.exportKey('PEM').decode('utf-8'))


generate_keys('A', RSA_KEY_SIZE)
generate_keys('B', RSA_KEY_SIZE)