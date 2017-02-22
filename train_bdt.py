import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from argparse import ArgumentParser
import tools, pandas, root_numpy
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
import sklearn
from pdb import set_trace
from files import files, base_feats, ivf_feats, epochs
from argparse import ArgumentParser
import os
from sklearn.externals import joblib

ntrees = 50
parser = ArgumentParser(description=__doc__)
parser.add_argument(
	'epoch', type=str, 
	choices=epochs,
	help='type of reconstruction'
	)
parser.add_argument(
	'--load', type=str, 
	help='load instead of training'
	)
args = parser.parse_args()
epoch = args.epoch

base_plotdir = '%s/trainplt_new_hiIP_rho_fakes' % epoch
if not os.path.isdir(base_plotdir):
	os.makedirs(base_plotdir)

data = pandas.DataFrame(
	root_numpy.root2array(
		files[epoch][:2], 'tree',
		)
	)

@np.vectorize
def score(history):
  if (history & 1):
    return 1
  elif (history & 2):
    return 1
  return 0

data['flavour'] = tools.flavours(data.history)
data['target'] = data.flavour >= 0

print 'Summary:'
print 'Total tracks:', data.shape[0]
print 'B     tracks:', (data.flavour == 2).sum() , ' (%.1f%%)' % ((data.flavour == 2).sum() *100./data.shape[0])
print 'C     tracks:', (data.flavour == 1).sum() , ' (%.1f%%)' % ((data.flavour == 1).sum() *100./data.shape[0])
print 'Light tracks:', (data.flavour == 0).sum() , ' (%.1f%%)' % ((data.flavour == 0).sum() *100./data.shape[0])
print 'Fake  tracks:', (data.flavour == -1).sum(), ' (%.1f%%)' % ((data.flavour == -1).sum()*100./data.shape[0])
print 'PU    tracks:', (data.flavour == -2).sum(), ' (%.1f%%)' % ((data.flavour == -2).sum()*100./data.shape[0])

data['ptratio'] = data.pt/data.Jet_pt
## base_feats.append('ptratio')
## ivf_feats.append('ptratio')

## print "Features used"
## for f in features:
## 	print '    ',f

data['default_cut'] = (data.pt > 1) & (data.chi2 < 5) & (data.dz < 17) & \
	 (data.dxy < 2) & (data.nHitPixel >= 1) & (data.nHitAll > 0) & \
	 (np.abs(data.dist) < 0.07) & (data.length < 5.0)

train, test = train_test_split(data, test_size=0.33, random_state=42)

#
# Training set masking
#
mask = train['IP2Dsig'] >= 2
train = train[mask]

if not args.load:
	clf = GradientBoostingClassifier(
		learning_rate=0.01, n_estimators=ntrees, subsample=0.8, random_state=13,
		max_features=len(base_feats), verbose=1,
		min_samples_leaf=int(0.01*len(train)),
		max_depth=5
		)
	clf_ivf = sklearn.base.clone(clf)
	clf_ivf.max_features = len(ivf_feats)
	
	clf.fit(train[base_feats], train.target)
	clf_ivf.fit(train[ivf_feats], train.target)
	
	joblib.dump(clf, '%s/full.pkl' % base_plotdir, compress=True)
	joblib.dump(clf_ivf, '%s/ivf.pkl' % base_plotdir, compress=True)
else:
	clf = joblib.load('%s/full.pkl' % args.load)
	clf_ivf = joblib.load('%s/ivf.pkl' % args.load)

test_pred     = clf.predict_proba(test[base_feats])[:, 1]
test_pred_ivf = clf_ivf.predict_proba(test[ivf_feats])[:, 1]
test['pred'] = test_pred
test['pred_ivf'] = test_pred_ivf

def make_plots(label, frame):
	plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	
	baseline_fpr = float(((frame.default_cut == 1) & (frame.target == 0)).sum()) #all
	baseline_fpr /= (frame.target == 0).sum()
	baseline_tpr  = float(((frame.default_cut == 1) & (frame.target == 1)).sum())
	baseline_tpr /=(frame.target == 1).sum()
	
	baseline_fake_fpr = float(((frame.default_cut == 1) & (frame.flavour == -1)).sum()) #fakes
	baseline_fake_fpr /= (frame.flavour == -1).sum()
	baseline_fake_tpr  = float(((frame.default_cut == 1) & (frame.target == 1)).sum())
	baseline_fake_tpr /=(frame.target == 1).sum()
	
	baseline_PU_fpr = float(((frame.default_cut == 1) & (frame.flavour == -2)).sum()) #PU
	baseline_PU_fpr /= (frame.flavour == -2).sum()
	baseline_PU_tpr  = float(((frame.default_cut == 1) & (frame.target == 1)).sum())
	baseline_PU_tpr /=(frame.target == 1).sum()
	
	#draw baseline point
	plt.plot([baseline_fpr], [baseline_tpr], label='baseline', markerfacecolor='red', marker='o', markersize=10)
	plt.plot([baseline_fake_fpr], [baseline_fake_tpr], markerfacecolor='red', marker='v', markersize=10)
	plt.plot([baseline_PU_fpr], [baseline_PU_tpr], markerfacecolor='red', marker='*', markersize=10)
	
	#plt.xscale('log')
	fakes = (frame.flavour == -1) | (frame.target == 1)
	pu    = (frame.flavour == -2) | (frame.target == 1)
	_, _, auc = tools.plot_roc(frame.pred, frame.target, True, label='full set (8 features)', color='blue')
	print "full roc auc", auc
	_, _, auc = tools.plot_roc(frame.pred[fakes.as_matrix()], frame.target[fakes], True, color='blue', ls='--')
	print "full roc auc (FAKES): ", auc
	_, _, auc = tools.plot_roc(frame.pred[pu.as_matrix()], frame.target[pu], True, color='blue', ls='-.')
	print "full roc auc (PU): ", auc
	
	_, _, auc = tools.plot_roc(frame.pred_ivf, frame.target, True, label='IVF set (6 features)', color='green')
	print "IVF roc auc", auc
	_, _, auc = tools.plot_roc(frame.pred_ivf[fakes.as_matrix()], frame.target[fakes], True, color='green', ls='--')
	print "IVF roc auc (FAKES):", auc
	_, _, auc = tools.plot_roc(frame.pred_ivf[pu.as_matrix()], frame.target[pu], True, color='green', ls='-.')
	print "IVF roc auc (PU):", auc
	plt.legend(loc='best')
	plt.title('Full set (dot, full lines), vs. fakes only (triangle, dashed), vs. PU (star, dot-dash)')
	plt.savefig('%s/%s_roc.png' % (base_plotdir, label))
	plt.clf()
	
	
	#draw baseline point
	sig_tot = (frame.target == 1).sum()
	sig_pass = float(((frame.target == 1) & (frame.default_cut == 1)).sum())
	eff = sig_pass/sig_tot
	fakes_pass = ((frame.flavour == -1) & (frame.default_cut == 1)).sum()
	pu_pass =  ((frame.flavour == -2) & (frame.default_cut == 1)).sum()
	plt.plot([eff], [sig_pass/(sig_pass+fakes_pass+pu_pass)], label='baseline', markerfacecolor='red', marker='o', markersize=10)
	plt.plot([eff], [sig_pass/(sig_pass+fakes_pass)], markerfacecolor='red', marker='v', markersize=10)
	plt.plot([eff], [sig_pass/(sig_pass+pu_pass)], markerfacecolor='red', marker='*', markersize=10)
	
	bdteff, bdtpur, cuts = tools.purity_vs_eff(frame.pred, frame.target, label='full set (8 features)', color='blue')
	info = zip(
		list(cuts),
		[abs(i-sig_pass/(sig_pass+fakes_pass+pu_pass)) for i in bdtpur]
		)
	cut = min(info, key=lambda x: x[1])[0]
	print "full roc auc", auc
	tools.purity_vs_eff(frame.pred[fakes.as_matrix()], frame.target[fakes], color='blue', ls='--')
	print "full roc auc (FAKES): ", auc
	tools.purity_vs_eff(frame.pred[pu.as_matrix()], frame.target[pu], color='blue', ls='-.')
	print "full roc auc (PU): ", auc
	
	tools.purity_vs_eff(frame.pred_ivf, frame.target, label='IVF set (6 features)', color='green')
	print "IVF roc auc", auc
	tools.purity_vs_eff(frame.pred_ivf[fakes.as_matrix()], frame.target[fakes], color='green', ls='--')
	print "IVF roc auc (FAKES):", auc
	tools.purity_vs_eff(frame.pred_ivf[pu.as_matrix()], frame.target[pu], color='green', ls='-.')
	print "IVF roc auc (PU):", auc
	plt.legend(loc='best')
	plt.title('Full set (dot, full lines), vs. fakes only (triangle, dashed), vs. PU (star, dot-dash)')
	plt.xlabel('efficiency')
	plt.ylabel('purity')
	plt.savefig('%s/%s_pur_eff.png' % (base_plotdir, label))
	plt.clf()
	return cut

cutval = make_plots('all', test)
mask = ((test.flavour < 0) | (test.flavour == 2))
make_plots('bonly', test[mask])
mask = ((test.flavour < 0) | (test.flavour == 1))
make_plots('conly', test[mask])
mask = (test.flavour <= 0)
make_plots('lonly', test[mask])

mask = (test.IP2Dsig > 2)
make_plots('all_highIP', test[mask])
mask = ((test.IP2Dsig > 2) & ((test.flavour < 0) | (test.flavour == 2)))
make_plots('bonly_highIP', test[mask])
mask = ((test.IP2Dsig > 2) & ((test.flavour < 0) | (test.flavour == 1)))
make_plots('conly_highIP', test[mask])
mask = ((test.IP2Dsig > 2) & (test.flavour <= 0))
make_plots('lonly_highIP', test[mask])

ranges = {
	'sip3dSig' : (-1000, 1000),
	'dzErr' : (0, 4),
	#'dist' : (-30, 0),
	'dxy' : (-0.5, 0.5),
	'IP2Dsig' : (-50, 100),
	'pPar' : (0, 200),
	'deltaR' : (0, 0.4),
	'pt' : (0, 50),
	'jetDistVal' : (-20, 0),
	'Jet_pt' : (0, 400),
	'dxyErr' : (0, 0.5),
	'ptRel' : (0, 5),
	'IP2Derr' : (0, 0.05),
	'decayLenVal' : (0, 100),
	'IP' : (-0.5, 0.5),
	'IPerr' : (0., 0.07),
	'dz' : (-1, 1),
	'IP2D' : (-0.2, 0.5),
	'sip2dVal' : (-20, 20),
	'p' : (0, 200),
	'length' : (-1, 5),
	'ptratio' : (0,0.5),
	'sip2dSig' : (-500, 500),
	'sip3dVal' : (-20, 20),
	'IPsig' : (-50, 150),
	'chi2' : (0, 6),
}
for column in base_feats:
	print 'plotting', column 
	btracks = test[column][test.pred > cutval]
	mM  = test[column].min(), test[column].max()
	if column in ranges:
		mM = ranges[column]
	plt.hist(
		btracks,
		color='b', alpha=0.3, range=mM, bins=50,
		histtype='stepfilled', normed=True,
		label='selected'
		)
	plt.xlabel(column)
	plt.ylabel("Arbitrary units")
	plt.legend(loc='best')
	plt.savefig('%s/%s.png' % (base_plotdir, column))
	plt.clf()


## tools.overtraining_plot(
## 	clf, 
## 	train[base_feats], train.target, 
## 	test[base_feats], test.target
## 	)
## plt.savefig('%s/overtraining_full.png' % base_plotdir)
## plt.clf()
## 
## tools.overtraining_plot(
## 	clf_ivf, 
## 	train[ivf_feats], train.target, 
## 	test[ ivf_feats], test.target
## 	)
## plt.savefig('%s/overtraining_ivf.png' % base_plotdir)
## plt.clf()
## 
#print feture importances
with open('%s/feats.raw_txt' % base_plotdir, 'w') as out:
	feats_vals = [i for i in zip(base_feats, clf.feature_importances_)]
 	feats_vals.sort(key=lambda x: x[1], reverse=True)
	for i in feats_vals:
		out.write('%s %f\n' % i)

with open('%s/feats_notv.raw_txt' % base_plotdir, 'w') as out:
	feats_vals = [i for i in zip(ivf_feats, clf_ivf.feature_importances_)]
 	feats_vals.sort(key=lambda x: x[1], reverse=True)
	for i in feats_vals:
		out.write('%s %f\n' % i)
## 
## 	
## #print decision fcn for B C and L
## X_train = train[base_feats]
## bdecision = clf.decision_function(X_train[train.flavour == 2]).ravel()
## cdecision = clf.decision_function(X_train[train.flavour == 1]).ravel()
## ldecision = clf.decision_function(X_train[train.flavour == 0]).ravel()
## decisions = [bdecision, cdecision, ldecision]
## mM = min(i.min() for i in decisions), max(i.max() for i in decisions)
## plt.hist(
## 	bdecision,
## 	color='b', alpha=0.3, range=mM, bins=30,
## 	histtype='stepfilled', normed=True,
## 	label='B tracks'
## 	)
## 
## plt.hist(
## 	cdecision,
## 	color='g', alpha=0.3, range=mM, bins=30,
## 	histtype='stepfilled', normed=True,
## 	label='C tracks'
## 	)
## 
## plt.hist(
## 	ldecision,
## 	color='r', alpha=0.3, range=mM, bins=30,
## 	histtype='stepfilled', normed=True,
## 	label='Other tracks'
## 	)
## plt.xlabel("BDT output")
## plt.ylabel("Arbitrary units")
## plt.legend(loc='best')
## plt.savefig('%s/byflavors.png' % base_plotdir)
## plt.clf()
## 
## #
## # Study effect of sorting
## #
## #add clf outpus
## data['BDTBase'] = clf.predict_proba(data[base_feats])[:, 1]
## data['BDTIVF']  = clf_ivf.predict_proba(data[ivf_feats])[:, 1]
## uuids = set(data.Jet_uuid)
## 
## class CumulativeDistro(object):
## 	def __init__(self, label='', color='k'):
## 		self.tpo = [0 for _ in range(50)] #true positive occurrences
## 		self.fpo = [0 for _ in range(50)] #false positive occurrences
## 		self.purity = [0. for _ in range(50)]
## 		self.fraction = [0. for _ in range(50)]
## 		self.zerosig = [0. for _ in range(50)]
## 		self.zerotrk = 0.
## 		self.label = label
## 		self.color = color
## 
## 	def add(self, targets):
## 		if len(self.tpo) < targets.shape[0]:
## 			diff = targets.shape[0] - len(self.tpo)
## 			default = self.tpo[-1] if self.tpo else 0
## 			self.tpo += [default for _ in range(diff)]
## 			default = self.fpo[-1] if self.fpo else 0
## 			self.fpo += [default for _ in range(diff)]
## 
## 		tpo = 0
## 		fpo = 0
## 		nsig = float(targets.sum())
## 		if not targets.shape[0]:
## 			self.zerotrk += 1
## 		for idx, val in enumerate(targets):
## 			if val == 1:
## 				tpo += 1
## 			else:
## 				fpo += 1
## 			self.tpo[idx] += tpo
## 			self.fpo[idx] += fpo
## 			self.purity[idx] += float(tpo)/(tpo+fpo)
## 			self.fraction[idx] += float(tpo)/nsig if nsig else 0.
## 			if not tpo:
## 				self.zerosig[idx] += 1
## 		
## 		for idx in range(targets.shape[0], len(self.tpo)):
## 			self.tpo[idx] += tpo
## 			self.fpo[idx] += fpo
## 			self.purity[idx] += float(tpo)/(tpo+fpo) if (tpo+fpo) else 0.
## 			self.fraction[idx] += float(tpo)/nsig if nsig else 0.
## 			if not tpo:
## 				self.zerosig[idx] += 1
## 
## 	def draw(self, signorm, bkgnorm, retain):
## 		tpo = [float(i)/signorm for i in self.tpo[:retain]]
## 		fpo = [float(i)/bkgnorm for i in self.fpo[:retain]]
## 		idx = [i+1 for i in range(retain)]
## 		plt.plot(
## 			idx, tpo, label=self.label, marker='o', 
## 			mec=self.color, mfc=self.color, mew=2, color=self.color
## 			)			 
## 		plt.xlabel("# kept tracks")
## 		plt.plot(
## 			idx, fpo, color=self.color, marker='o', 
## 			mec=self.color, mfc='white', mew=2, ls='--'
## 			)
## 
## 	def draw_purity_frac(self, njets, retain):
## 		pur = [float(i)/njets for i in self.purity[:retain]]
## 		fra = [float(i)/njets for i in self.fraction[:retain]]
## 		idx = [i+1 for i in range(retain)]
## 		plt.plot(
## 			idx, fra, label=self.label, marker='o', 
## 			mec=self.color, mfc=self.color, mew=2, color=self.color
## 			)			 
## 		plt.xlabel("# kept tracks")
## 		plt.plot(
## 			idx, pur, color=self.color, marker='o', 
## 			mec=self.color, mfc='white', mew=2, ls='--'
## 			)		
## 
## 	def draw_zsig(self, njets, retain):
## 		zsig = [float(i)/njets for i in self.zerosig[:retain]]
## 		idx = [i+1 for i in range(retain)]
## 		plt.plot(
## 			idx, zsig, label=self.label, marker='o', 
## 			mec=self.color, mfc=self.color, mew=2, color=self.color
## 			)			 
## 		plt.xlabel("# kept tracks")
## 
## 	def __repr__(self):
## 		return '  true positives : %s\n  false positives: %s' % (
## 			self.tpo, self.fpo)
## 
## cumulatives = {
## 	'BDTBase'  : CumulativeDistro('BDTBase' , 'r'),
## 	'BDTIVF'   : CumulativeDistro('BDTIVF'  , 'g'),
## 	'Sip2DSig' : CumulativeDistro('Sip2DSig', 'b'),
## 	'Default'  : CumulativeDistro('Default' , 'k'),
## }
## 
## slimmed_data = data[
## 	['Jet_uuid', 'BDTBase', 'BDTIVF', 
## 	 'IP2Dsig', 'target', 'default_cut']
## 	]
## grouped = slimmed_data.groupby('Jet_uuid')
## 
## for _, jet_data in grouped:
## 	cumulatives['BDTBase'].add(
## 		jet_data.sort_values('BDTBase', ascending=False).target
## 		)
## 	cumulatives['BDTIVF'].add(
## 		jet_data.sort_values('BDTIVF', ascending=False).target
## 		)
## 	cumulatives['Sip2DSig'].add(
## 		jet_data.sort_values('IP2Dsig', ascending=False).target
## 		)
## 	cumulatives['Default'].add(
## 		jet_data[jet_data.default_cut].sort_values(
## 			'IP2Dsig', ascending=False).target
## 		)
## 
## 
## nsignal = (slimmed_data.target == 1).sum()
## nbkg = (slimmed_data.target == 0).sum()
## njets = len(set(slimmed_data.Jet_uuid))
## 
## for label, cumul in cumulatives.iteritems():
## 	cumul.draw(nsignal, nbkg, 10)
## plt.legend(loc='best')
## plt.title(r'$\varepsilon_{sig}$ (solid), $\varepsilon_{bkg}$ (dashed)')
## plt.savefig('%s/acceptance_rates.png' % base_plotdir)
## plt.clf()
## 
## for label, cumul in cumulatives.iteritems():
## 	cumul.draw_purity_frac(njets, 10)
## plt.legend(loc='best')
## plt.title(r'$<\frac{N_{sig}(jet)}{N_{sig}^{TOT}(jet)}>$ (solid), $<\frac{N_{sig}(jet)}{N_{kept}(jet)}>$ (dashed)')
## plt.savefig('%s/retention_rates.png' % base_plotdir)
## plt.clf()
## 
## for label, cumul in cumulatives.iteritems():
## 	cumul.draw_zsig(njets, 10)
## plt.legend(loc='best')
## plt.ylabel('fraction of jets with no signal tracks')
## plt.savefig('%s/zerosig.png' % base_plotdir)
## plt.clf()
## 
## print 'Number of B/C jets without tracks: ', float(cumulatives['Default'].zerotrk)/njets
