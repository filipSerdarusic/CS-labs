from multiprocessing import Pipe, Process
import time
import random
import numpy as np

class Philosopher():

	def __init__(self, i, pipes):
		self.i = i
		self.pipes = pipes
		self.name = "Filozof " + str(i)
		self.clock = random.randint(1,30)
		self.N = len(pipes) + 1


	def participate(self):
		print('{} - sudjelujem na konferenciji'.format(self.name))
		time.sleep(random.uniform(0.1, 2))

		# Provjeri poruke - odgovori na zahtjeve

		# Zahtjev - ("zahtjev", participant_j, clock_j)
		# Odgovor - ("odgovor", participant_i, clock_j)

		for pipe in self.pipes:

			# Ako ima zahtjeva, odgovori na njih
			if (pipe.poll()):

				msg = pipe.recv()
				print('{} - primljena poruka: {}'.format(self.name, str(msg)))

				msg_type, j, clock_j = msg

				# Ažuriraj lokalni sat
				self.clock = max(self.clock, clock_j) + 1

				msg = ("odgovor", self.i, clock_j)
				pipe.send(msg)

				print('{} - poslana poruka: {}'.format(self.name, str(msg)))


	def eat(self):
		# Pošalji svima zahtjev
		print('\n{} - ŽELIM JESTI'.format(self.name))

		out_msg = ("zahtjev", self.i, self.clock)

		print('{} - šaljem svima: {}'.format(self.name, str(out_msg)))

		for pipe in self.pipes:
			pipe.send(out_msg)

		responses = np.array([False for _ in range(self.N)])
		responses[self.i] = True
		requests = []

		# Čekaj "odgovor" od svih ostalih
		while True:
			time.sleep(0.2)

			# Provjeri sve pristigle poruke ("odgovor"/"zahtjev")
			for pipe in self.pipes:
				if (pipe.poll()):
					msg = pipe.recv()
					
					msg_type, j, clock_j = msg

					# Ažuriraj lokalni sat
					self.clock = max(self.clock, clock_j) + 1
					print('{} - primljena poruka: {}'.format(
														self.name, str(msg)))

					# Provjeri tip poruke
					if msg_type == "odgovor":
						responses[j] = True
					
					else:
						# Poruka je "zahtjev"
						if clock_j < out_msg[2]:

							# Pristigli zahtjev je kasnije poslan
							# ... pošalji "odgovor" nakon kritičnog odsječka
							requests.append((pipe, ("odgovor",self.i,clock_j)))

						else:
							# Zahtjev ima prioritet, odmah pošalji odgovor
							msg = ("odgovor", self.i, clock_j)
							pipe.send(msg)
							print('{} - poslana poruka: {}'.format(
															self.name, str(msg)))

			# Provjeri jesu li stigli svi odgovori
			if (responses == True).all():
				break

		# Ulazak u kritični odsječak
		print('\n{} - ZA STOLOM\n'.format(self.name))
		time.sleep(3)

		# Odgovori svima koji čekaju "odgovor"
		for pipe, msg in requests:
			pipe.send(msg)


	def do(self):
		self.participate()
		self.eat()
		self.participate()


if __name__ == "__main__":
	
	N = 4
	processes = []

	# Kreiranje cjevovoda
	pipes = np.array((N**2)*[None]).reshape(N,N)

	for i in range(N):
		for j in range(i+1, N):
			p_ab, p_ba = Pipe()

			pipes[i, j] = p_ab
			pipes[j, i] = p_ba
	pipes = pipes.tolist()

	
	for i in range(N):
		pipes[i].remove(None)

	# Kreiranje procesa
	for i in range(N):
		processes.append(Process(target=Philosopher(i, pipes[i]).do))
	random.shuffle(processes)

	
	# Pokretanje
	for p in processes:
		p.start()