import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import subprocess
import math
import time
import os


threads = [1,2,4,8]
time_steps = [100, 500, 1000, 5000, 10000, 50000]
system_sizes = [64, 128, 256, 512, 1024, 2048, 4096, 8192]

def execute(filename, multi):
	if multi:
		m = "multiple_"
		e = "2"
		time_steps = [100]
		system_sizes = [64, 128, 256]
	else:
		m = ""
		e = "1"
		time_steps = [100, 500, 1000, 5000, 10000, 50000]
		system_sizes = [64, 128, 256, 512, 1024]

	target_name = "parallelized"

	indeces = pd.MultiIndex.from_product([time_steps, threads], names=["repetitions", "threads"])
	columns = pd.Series(system_sizes, name="system sizes")
	runtime = pd.DataFrame(columns=columns, index=indeces)

	# Compile target
	if os.system("g++ -fopenmp -o " + target_name + " exo_1." + e + "/" + m + "system_solver.cpp") != 0:
		exit(1)
	
	# Execute target and collect runtimes
	for size in system_sizes:
		print("System size: ", size)
		for nb_threads in threads:
			#change environmental variable OMP_NUM_THREADS to 1, 2...
			os.environ["OMP_NUM_THREADS"] = str(nb_threads)
			print("	threads set to: ", nb_threads)
			for nb_steps in time_steps:
				while True:
					print("		*time steps: ", nb_steps)
					time.sleep(0.5)
					#run the c++ executable with proper parameters
					output = subprocess.run(["./" + target_name,
								"exo_1." + e + "/input_data_exo_1." + e + "/" + m + "matrice_file_" + str(size) + ".csv", 
								str(nb_steps),
								"exo_1." + e + "/output_data_exo_1." + e + "/validation_" + m + "matrice_file_" + str(size) + "_" + str(nb_steps) + ".csv"
								],
						stdout=subprocess.PIPE, universal_newlines=True)
					if output.returncode == 0:
						runtime.loc[(nb_steps, nb_threads), size] = int(output.stdout[10:])
						break
		runtime.to_csv(m + filename + ".csv")
	
	if os.system("rm " + target_name) != 0:
		exit(1)


def strong(runtime):
	# Strong Scale UP
	for nb_steps in time_steps:
		for size in system_sizes:
			scores = []
			for nb_threads in threads:
				scores.append(runtime.loc[(nb_steps, nb_threads), size])
			plt.plot(scores, label= str(size) + " variables")
			if size == system_sizes[math.floor(len(system_sizes)/2)-1] or size == system_sizes[-1]:
				#plot results
				plt.xticks(range(len(threads)), threads)
				plt.legend()
				plt.title("Strong Scale Up Performance Evaluation\nfor " + str(nb_steps) + " steps")
				plt.yscale("log")
				plt.ylabel("Execution time")
				plt.xlabel("Number of threads")
				plt.grid(True)
				title = "graphs/strong_eval_" + str(nb_steps) + "_" + str(int(size/8)) + "-" + str(size) + ".pdf"
				plt.savefig(title)
				print(title)
				plt.cla()


def weak(runtime):
	# Weak Scale UP
	for nb_steps in time_steps:
		for i in [0,1]:	
			scores = []
			for nb_threads in threads:
				index = threads.index(nb_threads)
				size = system_sizes[i*len(threads) + index]
				scores.append(runtime.loc[(nb_steps, nb_threads), size])
			label = str(int(size/8)) + " / " + str(int(size/4)) + " / " + str(int(size/2)) + " / " + str(size)
			plt.plot(scores, label=label)
	
	
	
		#plot results
		plt.xticks(range(len(threads)), threads)
		plt.legend(title="System sizes")
		plt.title("Weak Scale Up Performance Evaluation\nfor " + str(nb_steps) + " steps")
		plt.yscale("log")
		plt.ylabel("Execution time")
		plt.xlabel("Number of threads")
		plt.grid(True)
		title = "graphs/weak_eval_" + str(nb_steps) + ".pdf"
		plt.savefig(title)
		print(title)
		plt.cla()


def plot_expected():
	#strong
	runtime = read("results_mean.csv", 5)
	scores = runtime.loc[(10000)][1024].tolist()
	scores = pd.Series(scores, threads)
	plt.plot(scores, 'm.-', label = "actual")
	f = scores[1]
	scores = [(f/x) for x in threads]
	scores = pd.Series(scores, threads)
	plt.plot(scores, 'k.--', label = "expected")
	plt.legend()
	plt.title("Strong Scale Up Performance Evaluation\n")
	plt.ylabel("Execution time")
	plt.xlabel("Number of threads")
	plt.grid(True)
	title = "expected_strong.pdf"
	plt.savefig(title)
	plt.cla()
	#limit
	x_cords = np.arange(1, 8, 0.2)
	y_cords = [1/x for x in x_cords]
	plt.plot(x_cords, y_cords, 'k--')
	plt.yticks([])
	plt.title("Strong Scale Up Performance In The Limit\n")
	plt.ylabel("Execution time")
	plt.xlabel("Number of threads")
	plt.savefig("limit.pdf")
	plt.cla()

	#weak
	runtime = read("results_mean.csv", 5)
	scores = []
	for th, size in zip(threads, system_sizes):
		scores.append(runtime.loc[(10000, th), size])
	scores = pd.Series(scores, threads)
	plt.plot(scores, 'c.-', label = "actual")
	scores = scores[1]
	scores = pd.Series(scores, threads)
	plt.plot(scores, 'k.--', label = "expected")
	plt.legend()
	plt.title("Weak Scale Up Performance Evaluation\n")
	plt.ylabel("Execution time")
	plt.xlabel("Number of threads")
	plt.grid(True)
	title = "expected_weak.pdf"
	plt.savefig(title)
	plt.cla()


def read(filename, n):
	return pd.read_csv(filename, index_col=[0,1], header=0, names=system_sizes[:n])


def get_mean(multi):
	if multi:
		m = "multiple_"
		n = 3
	else:
		m = ""
		n = 5

	li = []
	for i in range(4):
		df = read(m + "results" + str(i) + ".csv", n)
		li.append(df)

	frame = pd.concat(li, axis=0)
	return frame.mean(level=[0,1])


if __name__=="__main__":
	plot_expected()

