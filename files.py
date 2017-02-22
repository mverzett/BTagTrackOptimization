from glob import glob
epochs = {'81X', '80X'}
files = {
	'81X' : glob('81XBATCH_2017-02-22_08:18:57/*.root'),
	'80X' : glob('80XBATCH_2017-02-13_13:11:22/*.root'),
}
base_feats = ['pt', 'rho', 'chi2', 'dz', 'dxy', 'nHitPixel', 'nHitAll', 'length', 'dist']
ivf_feats  = ['pt', 'rho', 'chi2', 'dz', 'dxy', 'nHitPixel', 'nHitAll']
