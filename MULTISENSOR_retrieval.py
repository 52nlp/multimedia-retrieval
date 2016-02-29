<!-- MULTISENSOR_retrieval.py -->

import math as mh
import numpy as np
import heapq as hq
import sys
import scipy.spatial.distance as sdi
import operator
import string
v = 10

if(len(sys.argv) == 14) :
	textualSimilarityVectorFile = sys.argv[1]
	textualSimilarityMatrixFile = sys.argv[2]
	visFeaturesQueryFile  = sys.argv[3]
	visFeaturesFile  = sys.argv[4]
	visConceptsQueryFile  = sys.argv[5]
	visConceptsFile  = sys.argv[6]
	imagesListFile = sys.argv[7]
	imageQueryId = sys.argv[8]
	b1 = float(sys.argv[9])
	b2 = float(sys.argv[10])
	g1 = float(sys.argv[11])
	g2 = float(sys.argv[12])
	resultPath = sys.argv[13]


# Parameter setting
b3 = 1 - b1 - b2
g3 = 1 - g1 - g2
	
	
# Load file with ids
ImgIds = np.loadtxt(imagesListFile, dtype=int, ndmin =2)
	

######################### TEXTUAL MODALITY ##########################
# Matrix St and vector stq
stq = np.loadtxt(textualSimilarityVectorFile, ndmin =2)
St = np.loadtxt(textualSimilarityMatrixFile)


# Number of rows and columns of array
dim = St.shape
numrows = dim[0]
numcols = dim[1]


## Normalize (per row normalization) textual modality matrix and vector
maxSt = np.transpose(np.amax(St, axis=1) * 1.0 * np.ones((1, numcols)))
minSt = np.transpose(np.amin(St, axis=1) * 1.0 * np.ones((1, numcols)))
St_norm = ((St - minSt)*1.0)/(maxSt - minSt) 


maxStq = np.amax(stq, axis=1) * 1.0
minStq = np.amin(stq, axis=1) * 1.0
st_q_norm = ((stq - minStq)*1.0)/(maxStq - minStq) 


######################### VISUAL MODALITY ###########################
# Matrix Sv and query vector sv_q with features
Fvq = np.loadtxt(visFeaturesQueryFile, ndmin =2) 
Fv = np.loadtxt(visFeaturesFile)


# Estimating euclidean distances among all data
Dv = sdi.squareform(sdi.pdist(Fv))

# Estimating euclidean distances among query and all data
Fvq_repeat = np.repeat(Fvq, numrows, axis=0)
dvq_repeat = sdi.cdist(Fvq_repeat, Fv)
dvq = dvq_repeat[0,:]*np.ones((1, numcols))


# Get similarity matrix and vector for visual data using max function (also does normalization to [0,1])
maxDv = Dv.max() * 1.0
Sv_norm = 1.0 - Dv/maxDv

maxDvq = dvq.max() * 1.0
sv_q_norm = 1.0 - dvq/maxDvq


# Normalize (per row normalization) visual modality matrix and vector
maxSv_norm = np.transpose(np.amax(Sv_norm, axis=1) * 1.0 * np.ones((1, numcols)))
minSv_norm = np.transpose(np.amin(Sv_norm, axis=1) * 1.0 * np.ones((1, numcols)))
Sv_norm_tmp = ((Sv_norm - minSv_norm)*1.0)/(maxSv_norm - minSv_norm) 

maxSvq_norm = np.amax(sv_q_norm, axis=1) * 1.0
minSvq_norm = np.amin(sv_q_norm, axis=1) * 1.0
sv_q_norm_tmp = ((sv_q_norm - minSvq_norm)*1.0)/(maxSvq_norm - minSvq_norm) 

Sv_norm = Sv_norm_tmp
sv_q_norm = sv_q_norm_tmp


##################### VISUAL CONCEPT MODALITY #######################
# Matrix Sc and query vector sc_q with features
Fcq = np.loadtxt(visConceptsQueryFile, ndmin =2) 
Fc = np.loadtxt(visConceptsFile)


# Estimating euclidean distances among all data
Dc = sdi.squareform(sdi.pdist(Fc))

# Estimating euclidean distances among query and all data
Fcq_repeat = np.repeat(Fcq, numrows, axis=0)
dcq_repeat = sdi.cdist(Fcq_repeat, Fc)
dcq = dcq_repeat[0,:]*np.ones((1, numcols))


# Get similarity matrix and vector for visual concept data using max function (also does normalization to [0,1])
maxDc = Dc.max() * 1.0
Sc_norm = 1.0 - Dc/maxDc

maxDcq = dcq.max() * 1.0
sc_q_norm = 1.0 - dcq/maxDcq

# Normalize (per row normalization) visual modality matrix and vector
maxSc_norm = np.transpose(np.amax(Sc_norm, axis=1) * 1.0 * np.ones((1, numcols)))
minSc_norm = np.transpose(np.amin(Sc_norm, axis=1) * 1.0 * np.ones((1, numcols)))
Sc_norm_tmp = ((Sc_norm - minSc_norm)*1.0)/(maxSc_norm - minSc_norm) 

maxScq_norm = np.amax(sc_q_norm, axis=1) * 1.0
minScq_norm = np.amin(sc_q_norm, axis=1) * 1.0
sc_q_norm_tmp = ((sc_q_norm - minScq_norm)*1.0)/(maxScq_norm - minScq_norm) 

Sc_norm = Sc_norm_tmp
sc_q_norm = sc_q_norm_tmp


######################## MODALITY MERGING ############################
Cx = (1-b1-b2)*St_norm + b1* Sv_norm + b2 * Sc_norm
Cy = (1-b1-b2)*Sv_norm + b1* St_norm + b2 * Sc_norm
Cz = (1-b1-b2)*Sc_norm + b1* Sv_norm + b2 * St_norm

# Row normalization (sum per row = 1)
sumCx = np.sum(Cx, axis=1) * 1.0 * np.ones((1,numcols))
Px = Cx/(np.transpose(sumCx))

sumCy = np.sum(Cy, axis=1) * 1.0 * np.ones((1,numcols))
Py = Cy/(np.transpose(sumCy))

sumCz = np.sum(Cz, axis=1) * 1.0 * np.ones((1,numcols))
Pz = Cz/(np.transpose(sumCz))


######################### SCORE CALCULATION ##########################
# Getting v=10 bigger values from queries
if numrows<v:
	v = numrows
	
index_stq = hq.nlargest(v, xrange(numcols), st_q_norm.take)
index_svq = hq.nlargest(v, xrange(numcols), sv_q_norm.take)
index_scq = hq.nlargest(v, xrange(numcols), sc_q_norm.take)

Kst = np.zeros((1, numcols))
Ksv = np.zeros((1, numcols))
Ksc = np.zeros((1, numcols))
for x in range(0, v):
	istq = index_stq[x]
	isvq = index_svq[x]
	iscq = index_scq[x]
	Kst[0,istq] = st_q_norm[0,istq]
	Ksv[0,isvq] = sv_q_norm[0,isvq]
	Ksc[0,iscq] = sc_q_norm[0,iscq]

e = np.ones((numrows, 1))
x1 = np.dot(Kst, ((1-g1-g2) * Px + g1 * e * st_q_norm + g2 * e * sv_q_norm))
y1 = np.dot(Ksv, ((1-g1-g3) * Py + g1 * e * st_q_norm + g3 * e * sc_q_norm))
z1 = np.dot(Ksc, ((1-g2-g3) * Pz + g2 * e * sv_q_norm + g3 * e * sv_q_norm)) 

score = (1.0/6) * (st_q_norm + sv_q_norm + sc_q_norm + x1 + y1 + z1)


######################## FINAL PROCESSING ############################
# Transpose score array
scoreT = np.transpose(score)

# Concatenate imageIds and scores and sort them according to scores
result = np.concatenate((ImgIds,scoreT),axis=1)
result_sorted = sorted(result, key=operator.itemgetter(1), reverse=True)
result_sorted_toInt = np.asarray(result_sorted, dtype=int)


# Get only imageIds from sorted array
sorted_imageIds = result_sorted_toInt[:,0]


# Save result to file
#pathIndex = string.rfind(imagesListFile,'\\');
#resultFileName = imagesListFile[0:pathIndex] + "\\" + "result"+imageQueryId+".txt"
resultFileName = resultPath + "\\" + "result"+imageQueryId+".txt"
np.savetxt(resultFileName, sorted_imageIds, delimiter = ',', fmt='%i')
