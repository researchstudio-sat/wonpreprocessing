__author__ = 'hfriedrich'

import luigi
import luigi_evaluation_workflow

# This is the executed experiments script for the link prediction evaluation.
# It executes the luigi_evaluation_workflow with the parameters specified below.
#
# how to run (for installation see README.MD):
# - Extract the test data set 'testdataset_20141112' to folder C:/dev/temp
# - execute maven build (package) of this project to build the 'wonpreprocessing-1.0-SNAPSHOT-jar-with-dependencies.jar'
# - It is expected that your gate home is at 'C:/dev/GATE_Developer_8.0'
# - run this python script

TESTSET_FOLDER = 'C:/dev/temp/testdataset_20141112'
TENSOR_FOLDER = TESTSET_FOLDER + '/evaluation'
BASE_CONFIG = ['--lock-pid-dir', 'C:/dev/temp/luigi',
               '--local-scheduler', '--gatehome', 'C:/dev/GATE_Developer_8.0',
               '--inputfolder', TESTSET_FOLDER + '/data',
               '--tensorfolder', TENSOR_FOLDER,
               '--connections', TESTSET_FOLDER + '/connections.txt']
RESCAL_BASE_CONFIG = ['RESCALEvaluation'] + BASE_CONFIG
RESCAL_DEFAULT_PARAMS = ['--rank',  '500', '--threshold', '0.015']
RESCAL2_DEFAULT_PARAMS = ['--rank2',  '500', '--threshold2', '0.03']
COSINE_DEFAULT_PARAMS = ['--costhreshold', '0.5', '--costransthreshold', '0.0', '--wcosthreshold', '0.5', '--wcostransthreshold', '0.0']
COSINETRANNS_PARAMS = ['--costhreshold', '0.5', '--costransthreshold', '0.5', '--wcosthreshold', '0.5',
                          '--wcostransthreshold', '0.5']
ALL_DEFAULT_CONFIG = ['AllEvaluation'] + BASE_CONFIG + RESCAL_DEFAULT_PARAMS + RESCAL2_DEFAULT_PARAMS + \
                     COSINE_DEFAULT_PARAMS + ['--outputfolder', TENSOR_FOLDER + '/results/default']
COSINE_TRANS_CONFIG = ['CosineEvaluation'] + BASE_CONFIG + COSINETRANNS_PARAMS + \
                      ['--outputfolder', TENSOR_FOLDER + '/results/cosinetrans']
CATEGORY_CONFIG = ['CategoryEvaluation'] + BASE_CONFIG + ['--allneeds', TESTSET_FOLDER + '/allneeds.txt'] + ['--outputfolder', TENSOR_FOLDER + '/results/category']



# Rank evaluation: test the influence of the rank on the quality and
# performance of link prediction
def rank_eval():
    rank_threshold = [(25,[0.0001, 0.0003, 0.0005]),
                      (50,[0.001, 0.002, 0.003]),
                      (100,[0.005, 0.006, 0.007]),
                      (250,[0.01, 0.012, 0.015]),
                      (500,[0.012, 0.015, 0.02]),
                      (750,[0.015, 0.02, 0.025]),
                      (1000,[0.02, 0.025, 0.03])]
    for tuple in rank_threshold:
        rank = tuple[0]
        for threshold in tuple[1]:
            params = RESCAL_BASE_CONFIG + ['--outputfolder', TENSOR_FOLDER + '/results/rank'] + \
                     ['--rank', str(rank), '--threshold', str(threshold)]
            luigi.run(params)

def default_all_eval():
    luigi.run(ALL_DEFAULT_CONFIG)

def cosinetrans_eval():
    luigi.run(COSINE_TRANS_CONFIG)

def stemming_eval():
    luigi.run(ALL_DEFAULT_CONFIG + ['--stemming'] + ['--outputfolder', TENSOR_FOLDER + '/results/stemming'])

def content_slice_eval():
    luigi.run(RESCAL_BASE_CONFIG + RESCAL_DEFAULT_PARAMS + ['--content', '--additionalslices', 'subject.mtx content.mtx'] + \
              ['--outputfolder', TENSOR_FOLDER + '/results/content'])

def category_slice_eval():
    luigi.run(CATEGORY_CONFIG + RESCAL_DEFAULT_PARAMS)
    luigi.run(CATEGORY_CONFIG + ['--rank',  '500', '--threshold', '0.02'])
    luigi.run(CATEGORY_CONFIG + ['--rank',  '500', '--threshold', '0.03'])

def needtype_slice_eval():
    luigi.run(RESCAL_BASE_CONFIG + RESCAL_DEFAULT_PARAMS + ['--needtypeslice'] + ['--outputfolder', TENSOR_FOLDER + '/results/needtype'])

def maskrandom_eval():
    params = RESCAL_BASE_CONFIG + ['--outputfolder', TENSOR_FOLDER + '/results/maskrandom'] + ['--maskrandom']
    luigi.run(params + RESCAL_DEFAULT_PARAMS)
    luigi.run(params + ['--rank',  '500', '--threshold', '0.02'])
    luigi.run(params + ['--rank',  '500', '--threshold', '0.03'])

def connections_eval():
    connection_count = [0, 1, 2, 5, 10, 50, 100, 1000]
    for con in connection_count:
        luigi.run(RESCAL_BASE_CONFIG + RESCAL_DEFAULT_PARAMS + RESCAL2_DEFAULT_PARAMS + \
                  ['--maxconnections', str(con)] + ['--outputfolder', TENSOR_FOLDER + '/results/connections'])


if __name__ == '__main__':

    for i in range(3):
        default_all_eval()
        stemming_eval()
        category_slice_eval()
        needtype_slice_eval()
        maskrandom_eval()
        connections_eval()
        content_slice_eval()
        cosinetrans_eval()
        rank_eval()
