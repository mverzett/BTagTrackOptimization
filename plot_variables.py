#! /bin/env python

import root_numpy, pandas
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pdb import set_trace
from tools import flavours
from files import files, base_feats, epochs
from argparse import ArgumentParser
import os

parser = ArgumentParser(description=__doc__)
parser.add_argument(
	'epoch', type=str, 
	choices=epochs,
	help='type of reconstruction'
	)
args = parser.parse_args()
epoch = args.epoch

base_dir = '%s/plots_new_fakes' % epoch
if not os.path.isdir(base_dir):
	os.makedirs(base_dir)

data = pandas.DataFrame(
	root_numpy.root2array(
		files[epoch][:2], 'tree',
		)
	)
flavours = flavours(data.history)
data['ptratio'] = data.pt/data.Jet_pt
data['flavour'] = flavours

cols = set(data.columns) - {'history', 'score'}
nbins = 50
btrk = (flavours >= 2)
ctrk = (flavours == -1)
otrk = (flavours == -2)

print 'Summary:'
print 'Total tracks:', data.shape[0]
print 'B     tracks:', (data.flavour == 2).sum() , ' (%.0f%%)' % ((data.flavour == 2).sum() *100./data.shape[0])
print 'C     tracks:', (data.flavour == 1).sum() , ' (%.0f%%)' % ((data.flavour == 1).sum() *100./data.shape[0])
print 'Light tracks:', (data.flavour == 0).sum() , ' (%.0f%%)' % ((data.flavour == 0).sum() *100./data.shape[0])
print 'Fake  tracks:', (data.flavour == -1).sum(), ' (%.0f%%)' % ((data.flavour == -1).sum()*100./data.shape[0])
print 'PU    tracks:', (data.flavour == -2).sum(), ' (%.0f%%)' % ((data.flavour == -2).sum()*100./data.shape[0])

ranges = {}
## 	'sip3dSig' : (-1000, 1000),
## 	'dzErr' : (0, 4),
## 	#'dist' : (-30, 0),
## 	'dxy' : (-0.5, 0.5),
## 	'IP2Dsig' : (-50, 100),
## 	'pPar' : (0, 200),
## 	'deltaR' : (0, 0.4),
## 	'pt' : (0, 50),
## 	'jetDistVal' : (-20, 0),
## 	'Jet_pt' : (0, 400),
## 	'dxyErr' : (0, 0.5),
## 	'ptRel' : (0, 5),
## 	'IP2Derr' : (0, 0.05),
## 	'decayLenVal' : (0, 100),
## 	'IP' : (-0.5, 0.5),
## 	'IPerr' : (0., 0.07),
## 	'dz' : (-1, 1),
## 	'IP2D' : (-0.2, 0.5),
## 	'sip2dVal' : (-20, 20),
## 	'p' : (0, 200),
## 	'length' : (-1, 5),
## 	'ptratio' : (0,0.5),
## 	'sip2dSig' : (-500, 500),
## 	'sip3dVal' : (-20, 20),
## 	'IPsig' : (-50, 150),
## 	'chi2' : (0, 6),
## }
for column in cols:
	print 'plotting', column 
	btracks = data[column][btrk]
	ctracks = data[column][ctrk]
	otracks = data[column][otrk]	

	mM  = data[column].min(), data[column].max()
	if column in ranges:
		mM = ranges[column]
	plt.hist(
		btracks,
		color='b', alpha=0.3, range=mM, bins=nbins,
		histtype='stepfilled', normed=True,
		label='Real tracks'
		)
	plt.hist(
		ctracks,
		color='g', alpha=0.3, range=mM, bins=nbins,
		histtype='stepfilled', normed=True,
		label='Fake tracks'
		)
	plt.hist(
		otracks,
		color='r', alpha=0.3, range=mM, bins=nbins,
		histtype='stepfilled', normed=True,
		label='PU tracks'
		)
	plt.xlabel(column)
	plt.ylabel("Arbitrary units")
	plt.legend(loc='best')
	plt.savefig('%s/%s.png' % (base_dir, column))
	plt.clf()


#
# Correlation matrix
#
corrmat = data[base_feats][btrk].corr(method='pearson', min_periods=1)
fig, ax1 = plt.subplots(ncols=1, figsize=(12,10))
   
opts = {'cmap': plt.get_cmap("RdBu"),'vmin': -1, 'vmax': +1}
heatmap1 = ax1.pcolor(corrmat, **opts)
plt.colorbar(heatmap1, ax=ax1)

cat_ext = ''
ax1.set_title("Correlation Matrix ")

labels = corrmat.columns.values
for ax in (ax1,):
	# shift location of ticks to center of the bins
	ax.set_xticks(np.arange(len(labels))+0.5, minor=False)
	ax.set_yticks(np.arange(len(labels))+0.5, minor=False)
	ax.set_xticklabels(labels, minor=False, ha='right', rotation=70)
	ax.set_yticklabels(labels, minor=False)
        
plt.tight_layout()
plt.savefig('%s/Correlation.png' % base_dir)
plt.clf()

#
# Scatter combination
#
## import itertools
## for xval, yval in itertools.combinations(base_feats, 2):
## 	print 'plotting', xval, yval
## 	btracks = data[[xval, yval]][btrk]
## 	ctracks = data[[xval, yval]][ctrk]
## 	otracks = data[[xval, yval]][otrk]	
## 	## set_trace()
## 	## mM  = data[column].min(), data[column].max()
## 	## if column in ranges:
## 	## 	mM = ranges[column]
## 	plt.scatter(
## 		btracks[xval], btracks[yval],
## 		color='b', marker='o', alpha=0.2, linewidths=0,
## 		label='B tracks'
## 		)
## 	plt.scatter(
## 		ctracks[xval], ctracks[yval],
## 		color='g', marker='o', alpha=0.2, linewidths=0,
## 		label='C tracks'
## 		)
## 	plt.scatter(
## 		otracks[xval], otracks[yval],
## 		color='r', marker='o', alpha=0.2, linewidths=0,
## 		label='Other tracks'
## 		)
## 	plt.xlabel(xval)
## 	plt.ylabel(yval)
## 	plt.legend(loc='best')
## 	plt.savefig('%s/%s_%s.png' % (base_dir, xval, yval))
## 	plt.clf()
