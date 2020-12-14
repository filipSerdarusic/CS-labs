from multiprocessing import Queue
from threading import Thread
import random
import time


def posjetitelj(K, queue_in, queue_out, queue_f):

	for _ in range(3):

		time.sleep(random.uniform(0.1, 2))
		print("Posjetitelj {}: Želim se voziti!".format(K))
		queue_out.put('Želim se voziti')

		# Čekaj poziv za sjesti - poruku "Sjedi"
		while(True):
			m = queue_in.get()
			if m == 'Sjedi':
				break
			queue_in.put(m)

		print('Sjeo posjetitelj {}'.format(K))

		# Čekaj kraj vožnje - poruku "Ustani"
		while(True):
			m = queue_in.get()
			if m == 'Ustani':
				break
			queue_in.put(m)
		
		print('Sišao posjetitelj {}'.format(K))

	print('\nPosjetitelj {} završio\n'.format(K))
	queue_f.put(K)


def vrtuljak():

	num_visitors = 8
	max_visitors = 4
	finished = 0

	queue_in = Queue()
	queue_out = Queue()
	queue_f = Queue()

	visitors = []
	for i in range(num_visitors):
		visitors.append(Thread(target=posjetitelj, 
								args=(i, queue_in, queue_out, queue_f),
								daemon=True))

	# Pokreni dretve
	for thread in visitors:
		thread.start()

	while (finished < num_visitors):
		
		# Čekaj 4 posjetitelja
		while(queue_out.qsize() != 4):
			time.sleep(0.01)
		
		# Makni 4 zahtjeva iz reda
		count = 0
		while (count < max_visitors):
			m = queue_out.get()
			count += 1

		# Pošalji posjetiteljima poruku da sjednu
		for _ in range(4):
			queue_in.put('Sjedi')

		# Pokreni vrtuljak kada svi sjednu
		while(queue_in.qsize() != 0):
			time.sleep(0.01)

		print('\nPokrenuo vrtuljak!\n')
		time.sleep(random.uniform(1,3))
		print('\nVrtuljak zaustavljen!\n')

		for _ in range(4):
			queue_in.put('Ustani')
	
		while (queue_in.qsize() != 0):
			time.sleep(0.01)

		# Provjeri ima li posjetitelja koji su završili
		while(queue_f.qsize() != 0):
			K = queue_f.get()
			finished += 1

	print('Vrtuljak završio s radom')


if __name__ == "__main__":

	t = Thread(target=vrtuljak)
	t.start()
	t.join()