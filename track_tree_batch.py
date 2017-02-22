from argparse import ArgumentParser
from glob import glob
import os
import time
import stat

parser = ArgumentParser()
parser.add_argument('inputdir')
parser.add_argument('--split', type=int, default=5)
parser.add_argument('--prefix', default='')
args = parser.parse_args()

infiles = glob('%s/*.root' % args.inputdir)
print 'found', len(infiles), 'files'

infiles = [i.replace('/eos/uscms',  'root://cmseos.fnal.gov/') for i in infiles]
split = [
	[name for idx, name in enumerate(infiles) if idx % args.split == modval]
	for modval in range(args.split)
]

batch_id = '%sBATCH_%s' % (args.prefix, time.strftime("%Y-%m-%d_%H:%M:%S", time.gmtime()))
os.mkdir(batch_id)

batch_sh = '%s/batch.sh' % batch_id
with open(batch_sh, 'w') as batch:
	batch.write('''#!/bin/bash
WORKINGDIR=$PWD
cd {0}
eval `scramv1 runtime -sh`
cd $WORKINGDIR

make_track_tree $@ 

exitcode=$? 
echo "exit code: "$exitcode
exit $exitcode '''.format(os.environ['CMSSW_BASE']))

st = os.stat(batch_sh)
os.chmod(batch_sh, st.st_mode | stat.S_IEXEC)

with open('%s/condor.jdl' % batch_id, 'w') as jdl:
	jdl.write('''
universe = vanilla
Executable = batch.sh
Should_Transfer_Files = YES
WhenToTransferOutput = ON_EXIT
request_memory = 5000
''')
	for idx, chunk in enumerate(split):
		out = 'track_tree_%d.root' % idx
		jdl.write('''

Output = con_{IDX}.stdout
Error = con_{IDX}.stderr
Log = con_{IDX}.log
Arguments = {output} {inputs}
Queue'''.format(IDX=idx, output=out, inputs=' '.join(chunk))
							)

