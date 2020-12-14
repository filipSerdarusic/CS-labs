import base64
from Crypto import Random
from Crypto.PublicKey import RSA
from Crypto.Cipher import AES, DES3
from Crypto.Hash import SHA256, SHA512
from Crypto.Signature import PKCS1_v1_5

import generate_keys
from digital_envelope import DigitalEnvelope

# Public and Private Key files
# Keys are created in generate_keys.py

PUBLIC_KEY_A = 'public_key_A.pem'
PRIVATE_KEY_A = 'private_key_A.pem'

PUBLIC_KEY_B = 'public_key_B.pem'
PRIVATE_KEY_B = 'private_key_B.pem'


'''
CONFIGURATION

cipher: AES, DES3

mode:
	AES.MODE_ECB, AES.MODE_CBC
	DES3.MODE_ECB, DES3.MODE_CBC

cipher_key_size:
	AES  - 16, 24, 32
	DES3 - 16, 24

hash_function: SHA256, SHA512

'''

config = {
		'cipher' : AES,
		'mode' : AES.MODE_ECB,
		'cipher_key_size' : 32,
		'block_size' : 32,
		'hash_function' : SHA256
		}


# Import public keys
public_key_A = RSA.importKey(open('public_key_A.pem', 'r').read())
public_key_B = RSA.importKey(open('public_key_B.pem', 'r').read())


# Generate symmetric key for AES/3-DES encrypting
key = Random.get_random_bytes(config['cipher_key_size'])


# Creating Digital Envelope
envelope = DigitalEnvelope(key, config)


print("Signing the envelope...\n")

# Signing the envelope using sender's private key (A)
signature = envelope.sign(PRIVATE_KEY_A)


print("Veryfing digital signature...\n")

# Verifying the signature using sender's public key (A)
if envelope.verify_signature(signature, PUBLIC_KEY_A):
	print("Digital signature is valid.\n")
else:
	print("Digital signature is not valid.\n")


# Message to be communicated
message = 'This is the message.'
print("Message:", message, '\n')


# Encrypting the message using AES/3-DES symmetric key
message_enc = envelope.encrypt(message)
print("Encrypted message:", message_enc, "\n")


# Encrypting the AES/3-DES key using reciever's public key (B)
key_enc = envelope.encrypt_key(PUBLIC_KEY_B)


# Getting private_key_B
private_key = RSA.importKey(open('private_key_B.pem', 'r').read())


# Decrypting the encyrpted symmetric key using reciever's private key (B)
key_decrypted = private_key.decrypt(key_enc)


# Decrypting the message using AES/3-DES symmetric key
message_decrypted = envelope.decrypt(message_enc)
print("Decrypted message:", message_decrypted)