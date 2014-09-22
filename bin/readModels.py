#!/usr/bin/env pypy
import pickle
import sys
import glob
import time
from inference import *
from bootstrap import bootstrap


def perpGain(r1, r2):
	return (r1 - r2) / (r1 - 1)


def llGain(r1, r2):
	return (math.exp(r2 - r1) - 1)


def avg(l):
	s = 0
	n = 0
	for x in l:
		s += x
		n += 1
	return float(s) / n if n else 0


# TESTED_MODEL_PAIRS = ['UBM', 'EB_UBM', 'EB_UBM-IA']
TESTED_MODEL_PAIRS = ['MDBNvsDBN']

if 'RBP' in TESTED_MODEL_PAIRS:
	import scipy
	import scipy.optimize

MODEL_CONSTRUCTORS = {
	'DBN': (lambda: DbnModel((0.9, 0.9, 0.9, 0.9)), lambda: DbnModel((1.0, 0.9, 1.0, 0.9), ignoreIntents=False, ignoreLayout=False)),
	'MDBNvsDBN':  (lambda: MouseDbnModel((0.9, 0.9, 0.9, 0.9),0.5), lambda: DbnModel((0.9, 0.9, 0.9, 0.9))),
	'DBNvsMDBN':  (lambda: DbnModel((0.9, 0.9, 0.9, 0.9)),lambda: MouseDbnModel((0.9, 0.9, 0.9, 0.9),0.5)),
	'UBMvsDBN': (UbmModel, lambda: DbnModel((0.9, 0.9, 0.9, 0.9))),
	'UBM': (UbmModel, lambda: UbmModel(ignoreIntents=False, ignoreLayout=False)),
	'EB_UBM': (UbmModel, EbUbmModel, lambda: EbUbmModel(ignoreIntents=False, ignoreLayout=False)),
	'DCM': (DcmModel, lambda: DcmModel(ignoreIntents=False, ignoreLayout=False)),
	#'RBP': (SimplifiedRbpModel, lambda: RbpModel(ignoreIntents=False, ignoreLayout=False))
}



if __name__ == '__main__':
	#test path; out path ; model path ; model name
	if len(sys.argv) != 5:
		print >>sys.stderr, 'Usage: {0:s} directory_with_files'.format(sys.argv[0])
		sys.exit(1)
	out_file = sys.argv[2]
	#Dbn
	gamma = 0.9
	
	#UBM
	#print('UBM')
	
	total_time = 0.0
	
	interestingFiles = sorted(glob.glob(sys.argv[1] + '/*'))
	N = len(interestingFiles) // 2
	out = open(out_file,'w')
	print('out_file : ' + out_file)
	for fileNumber in xrange(N):
		models = []
		for i in xrange(11):
			pkl_file = open(sys.argv[3] + sys.argv[4] + '_' + str(i), 'rb')
			m = pickle.load(pkl_file)
			pkl_file.close()
			models.append(m)
		trainFile = interestingFiles[2 * fileNumber]
		print('trainFile : '+trainFile)
		out.write('trainFile : '+trainFile + '\n')
		testFile = interestingFiles[2 * fileNumber + 1]
		print('testFile : ' + testFile)
		out.write('testFile : ' + testFile + '\n')
		readInput = InputReader(False)
		trainSessions = readInput(open(trainFile))
		testSessions = readInput(open(testFile))
		res = []
		for idx, model in enumerate(models):
			start = time.time()
			m = model
			currentResult = m.test(testSessions, reportPositionPerplexity=True)
			res.append(currentResult)
			print >>sys.stderr, float(fileNumber) / N, idx, currentResult
			print("Model " + str(idx) + ":")
			out.write(str(idx)+'\n')
			print(currentResult)
			out.write(str(currentResult)+'\n')
			del m
			end = time.time()
			total_time += end-start
			print("Use Time = " + str(end-start))
			out.write("Use Time = " + str(end-start) + "\n")
		standardRes = res[0]
		diffLL = []
		diffPerplexity = []
		for i in xrange(len(res)):
			diffLL.append(res[i][0] - standardRes[0])
			diffPerplexity.append(standardRes[1] - res[i][1])
		print("LL Gains:")
		print(diffLL)
		print("Perplexity Gains:")
		print(diffPerplexity)	
	
		print("Total Use Time = " + str(total_time))

