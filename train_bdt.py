import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from argparse import ArgumentParser
import tools, pandas, root_numpy
from sklearn.ensemble import GradientBoostingClassifier
import sklearn
from pdb import set_trace

data = pandas.DataFrame(
	root_numpy.root2array(
		'track_tree.root', 'tree'
		)
	)

@np.vectorize
def score(history):
  if (history & 1):
    return 1
  elif (history & 2):
    return 1
  return 0

data['target'] = score(data.history)
data['flavour'] = tools.flavours(data.history)
tagvars = {
	'ptRel',
	'pPar',
	'etaRel',
	'deltaR',
	'ptRatio',
	'pParRatio',
	'sip2dVal',
	'sip2dSig',
	'sip3dVal',
	'sip3dSig',
	'decayLenVal',
	'decayLenSig',
	'jetDistVal',
	'jetDistSig',	
	#IP Stuff
	'IP2Dsig',
	'IP2Derr',
	'IP2D',
	'IP',
	'IPerr',
	'IPsig',
	#SV Stuff
	'isfromV0',
	'isfromSV',
	'SVweight',
}
geninfo = {
	'target', 'Jet_phi', 'Jet_eta', 
	'lheFilter', 'quality', 'category', 
	'charge', 'Jet_ncHadrons', 'phi', 
	'eta', 'Jet_nbHadrons', 'Jet_flavour', 
	'mcweight', 'history', 'flavour'
	}

features = list(
	set(data.columns) - geninfo
	)
feats_notv = list(set(features) - tagvars)
base_feats = ['pt', 'chi2', 'dz', 'dxy', 'nHitPixel', 'nHitAll', 'length', 'dist',]
nonjet = list(set([
	'pt', 'chi2', 'dz', 'dxy', 'nHitPixel', 'nHitAll', 
	'dxyErr', 'dzErr', 'ndof', 'nHitTOB', 'nHitTIB', 'nPV', 'nHitAll', 
	'nInactiveHits', 'nHitTID', 'isHitL1', 'nHitTEC', 'dxyErr', 'tight', 
	'nHitPixel', 'nHitPXB', 'nSiLayers', 'nPxLayers', 'loose', 'rho', 
	'nHitPXF', 'nHitStrip', 'highpurity', 'discarded', 'p', 'nLostHits'
	]))

## print "Features used"
## for f in features:
## 	print '    ',f

train, test = train_test_split(data, test_size=0.33, random_state=42)

default_cut = (test.pt > 1) & (test.chi2 < 5) & (test.dz < 17) & \
	 (test.dxy < 2) & (test.nHitPixel >= 1) & (test.nHitAll > 0) & \
	 (np.abs(test.dist) < 0.07) & (test.length < 5.0)
baseline_fpr = float((default_cut & (test.target == 0)).sum()) #nfakes
baseline_fpr /= (test.target == 0).sum()
baseline_tpr  = float((default_cut & (test.target == 1)).sum())
baseline_tpr /=(test.target == 1).sum()

clf = GradientBoostingClassifier(
	learning_rate=0.01, n_estimators=100, subsample=0.8, random_state=13,
	max_features=len(features), verbose=1,
	min_samples_leaf=int(0.01*len(train)),
	max_depth=5
)
clf_notv = sklearn.base.clone(clf)
clf_notv.max_features = len(feats_notv)
clf_base = sklearn.base.clone(clf)
clf_base.max_features = len(base_feats)
clf_nonj = sklearn.base.clone(clf)
clf_nonj.max_features = len(nonjet)

clf.fit(train[features], train.target)
clf_notv.fit(train[feats_notv], train.target)
clf_base.fit(train[base_feats], train.target)
clf_nonj.fit(train[nonjet], train.target)

test_pred      = clf.predict_proba(test[features])[:, 1]
test_pred_notv = clf_notv.predict_proba(test[feats_notv])[:, 1]
test_pred_base = clf_base.predict_proba(test[base_feats])[:, 1]
test_pred_nonj = clf_nonj.predict_proba(test[nonjet])[:, 1]

#draw random luck
plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

#draw baseline point
plt.plot([baseline_fpr], [baseline_tpr], label='baseline', markerfacecolor='red', marker='o', markersize=12)
#plt.xscale('log')
_, _, auc = tools.plot_roc(test_pred, test.target, True, label='full training', color='blue')
print "full roc auc", auc
_, _, auc = tools.plot_roc(test_pred_notv, test.target, True, label='no tagvar training', color='green')
print "partial roc auc", auc
_, _, auc = tools.plot_roc(test_pred_base, test.target, True, label='baseline vars', color='red')
print "base roc auc", auc
_, _, auc = tools.plot_roc(test_pred_nonj, test.target, True, label='nonjet vars', color='magenta', linestyle='dashed')
print "nonjet roc auc", auc
plt.legend(loc='best')
plt.savefig('trainplt/roc.png')
plt.clf()

tools.overtraining_plot(
	clf, 
	train[features], train.target, 
	test[features], test.target
	)
plt.savefig('trainplt/overtraining_full.png')
plt.clf()

tools.overtraining_plot(
	clf_notv, 
	train[feats_notv], train.target, 
	test[ feats_notv], test.target
	)
plt.savefig('trainplt/overtraining_notv.png')
plt.clf()

tools.overtraining_plot(
	clf_base, 
	train[base_feats], train.target, 
	test[ base_feats], test.target
	)
plt.savefig('trainplt/overtraining_base.png')
plt.clf()

#print feture importances
with open('trainplt/feats.raw_txt', 'w') as out:
	feats_vals = [i for i in zip(features, clf.feature_importances_)]
 	feats_vals.sort(key=lambda x: x[1], reverse=True)
	for i in feats_vals:
		out.write('%s %f\n' % i)

with open('trainplt/feats_notv.raw_txt', 'w') as out:
	feats_vals = [i for i in zip(feats_notv, clf_notv.feature_importances_)]
 	feats_vals.sort(key=lambda x: x[1], reverse=True)
	for i in feats_vals:
		out.write('%s %f\n' % i)

with open('trainplt/feats_base.raw_txt', 'w') as out:
	feats_vals = [i for i in zip(base_feats, clf_base.feature_importances_)]
 	feats_vals.sort(key=lambda x: x[1], reverse=True)
	for i in feats_vals:
		out.write('%s %f\n' % i)
	
#print decision fcn for B C and L
X_train = train[features]
bdecision = clf.decision_function(X_train[train.flavour == 2]).ravel()
cdecision = clf.decision_function(X_train[train.flavour == 1]).ravel()
ldecision = clf.decision_function(X_train[train.flavour == 0]).ravel()
decisions = [bdecision, cdecision, ldecision]
mM = min(i.min() for i in decisions), max(i.max() for i in decisions)
plt.hist(
	bdecision,
	color='b', alpha=0.3, range=mM, bins=30,
	histtype='stepfilled', normed=True,
	label='B tracks'
	)

plt.hist(
	cdecision,
	color='g', alpha=0.3, range=mM, bins=30,
	histtype='stepfilled', normed=True,
	label='C tracks'
	)

plt.hist(
	ldecision,
	color='r', alpha=0.3, range=mM, bins=30,
	histtype='stepfilled', normed=True,
	label='Other tracks'
	)
plt.xlabel("BDT output")
plt.ylabel("Arbitrary units")
plt.legend(loc='best')
plt.savefig('trainplt/byflavors.png')
plt.clf()



