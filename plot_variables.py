import root_numpy, pandas
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pdb import set_trace
from tools import flavours

data = pandas.DataFrame(
	root_numpy.root2array(
		'track_tree.root', 'tree'
		)
	)
flavours = flavours(data.history)

cols = set(data.columns) - {'history', 'score'}
nbins = 50
btrk = (flavours == 2)
ctrk = (flavours == 1)
otrk = (flavours == 0)
print 'Summary:'
print 'Total tracks:', data.shape[0]
print 'B tracks:', btrk.sum()
print 'C tracks:', ctrk.sum()
print 'Other tracks:', otrk.sum()
ranges = {
	'sip3dSig' : (-1000, 1000),
	'dzErr' : (0, 4),
	'dist' : (-30, 0),
	'dxy' : (-20, 20),
	'IP2Dsig' : (-1000, 1000),
	'pPar' : (0, 200),
	'deltaR' : (0, 0.4),
	'pt' : (0, 50),
	'jetDistVal' : (-20, 0),
	'Jet_pt' : (0, 400),
	'dxyErr' : (0, 0.5),
	'ptRel' : (0, 5),
	'IP2Derr' : (0, 0.5),
	'decayLenVal' : (0, 100),
	'IP' : (-20, 20),
	'IPerr' : (0, 2),
	'dz' : (-20, 20),
	'IP2D' : (-20, 20),
	'sip2dVal' : (-20, 20),
	'p' : (0, 200),
	'length' : (-10, 100),
	'ptRatio' : (0,0.5),
	'sip2dSig' : (-500, 500),
	'sip3dVal' : (-20, 20),
	'IPsig' : (-1000, 1000),
	'chi2' : (0, 15),
}
for column in ['chi2']:
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
		label='B tracks'
		)
	plt.hist(
		ctracks,
		color='g', alpha=0.3, range=mM, bins=nbins,
		histtype='stepfilled', normed=True,
		label='C tracks'
		)
	plt.hist(
		otracks,
		color='r', alpha=0.3, range=mM, bins=nbins,
		histtype='stepfilled', normed=True,
		label='Other tracks'
		)
	plt.xlabel(column)
	plt.ylabel("Arbitrary units")
	plt.legend(loc='best')
	plt.savefig('plots/%s.png' % column)
	plt.clf()
