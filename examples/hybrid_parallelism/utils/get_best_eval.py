import sys
import glob
import numpy as np

dir = sys.argv[1]

file_dirs = glob.glob(dir + '/*')
print(file_dirs)


best_scores = []
for file_dir in file_dirs:
	f = file_dir + '/workerlog.0'
	with open(f) as r:
		best_score = 0.0
		best_step = 0
		for line in r:
			if line.startswith('[evaluation]'):
				lines = line.strip().split(' ')
				step = int(lines[2][:-1])
				score = float(lines[7])
				if score > best_score:
					best_score = score
					best_step = step
		print(f, best_score, best_step)
		best_scores.append(best_score)
print([float(str(b)[:8])*100 for b in best_scores])

best_scores.sort()
print("{} times mean: {:.3f}, median: {:.3f}".format(len(best_scores), np.mean(best_scores)*100, np.median(best_scores)*100))
