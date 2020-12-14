import base64
from Crypto import Random
from Crypto.PublicKey import RSA
from Crypto.Cipher import AES, DES3
from Crypto.Hash import SHA256, SHA512
from Crypto.Signature import PKCS1_v1_5


class DigitalEnvelope():

	def __init__(self, key, config):

		'''
		key 			- AES/3-DES symmetric key
		cipher 			- AES/3-DES
		hash_function 	- SHA256, SHA512
		bs 				- block_size 
		'''

		self.key = key
		self.cipher = config['cipher'].new(key, mode=config['mode'])
		self.hash_function = config['hash_function']
		self.bs = config['block_size']


	def encrypt(self, message):
		
		''' Encrypting the message using AES/3-DES symmetric key '''

		encypted = self.cipher.encrypt(self.pad(message))
		encypted = base64.b64encode(encypted).decode('utf-8')
		return encypted
	

	def decrypt(self, encrypted):
		
		''' Decrypting the message using AES/3-DES symmetric key '''

		encrypted = base64.b64decode(encrypted)
		message = self.cipher.decrypt(encrypted)
		return self.unpad(message).decode('utf-8')
	

	def encrypt_key(self, public_key):

		''' Encrypting the symmetric key with reciever's public key (B) '''

		public_key = RSA.importKey(open(public_key, 'r').read())
		key_enc = public_key.encrypt(self.key, K=self.bs)
		key_enc = base64.b64encode(key_enc[0])
		return key_enc


	def sign(self, private_key, message=None):

		''' Signing the message using sender's private key (A) '''

		if message is None:
			message = 'To be signed'

		key = RSA.importKey(open(private_key, 'r').read())
		signer = PKCS1_v1_5.new(key)

		h = self.generate_hash(message)
		signature = signer.sign(h)
		return signature


	def verify_signature(self, signature, public_key, message=None):

		''' Verifying the signature using sender's public key (A) '''

		if message is None:
			message = 'To be signed'
	
		h = self.generate_hash(message)
		public_key = RSA.importKey(open(public_key, 'r').read())
	
		verifier = PKCS1_v1_5.new(public_key)
		return verifier.verify(h, signature)


	def generate_hash(self, data):
		if not isinstance(data, bytes):
			data = data.encode('utf-8')
		return self.hash_function.new(data)


	def pad(self, s):
		return s + (self.bs - len(s) % self.bs) * chr(self.bs - len(s) % self.bs)


	def unpad(self,s):
		return s[:-ord(s[len(s)-1:])]
