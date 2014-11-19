__author__ = 'hfriedrich'

import luigi
import luigi_evaluation_workflow

# This is the executed experiments script for the link prediction evaluation on the 'testdataset_20141112'.
# It executes the luigi_evaluation_workflow with the parameters specified below.
#
# how to run (for installation see README.MD):
# - Extract the test data set 'testdataset_20141112' to folder C:/dev/temp
# - execute maven build (package) of this project to build the 'wonpreprocessing-1.0-SNAPSHOT-jar-with-dependencies.jar'
# - It is expected that your gate home is at 'C:/dev/GATE_Developer_8.0'
# - run this python script

TESTSET_FOLDER = 'C:/dev/temp/testdataset_20141112'
OUTPUT_FOLDER = TESTSET_FOLDER + '/evaluation'
BASE_CONFIG = ['--lock-pid-dir', 'C:/dev/temp/luigi',
               '--local-scheduler', '--gatehome', 'C:/dev/GATE_Developer_8.0',
               '--inputfolder', TESTSET_FOLDER + '/data',
               '--connections', TESTSET_FOLDER + '/connections.txt']
RESCAL_BASE_CONFIG = ['RESCALEvaluation'] + BASE_CONFIG
RESCAL_DEFAULT_PARAMS = ['--rank',  '500', '--threshold', '0.02']
RESCAL2_DEFAULT_PARAMS = ['--rank2',  '500', '--threshold2', '0.05']
COSINE_DEFAULT_PARAMS = ['--costhreshold', '0.5', '--costransthreshold', '0.0', '--wcosthreshold', '0.5', '--wcostransthreshold', '0.0']

# evaluate all algorithms in their default configuration
def default_all_eval():
    params = ['AllEvaluation'] + BASE_CONFIG + RESCAL_DEFAULT_PARAMS + RESCAL2_DEFAULT_PARAMS + \
             COSINE_DEFAULT_PARAMS + ['--outputfolder', OUTPUT_FOLDER + '/results/default'] + \
             ['--tensorfolder', OUTPUT_FOLDER + '/tensor']
    luigi.run(params)

# evaluate the influence of the rank on the quality and performance of link prediction
def rank_eval():
    rank_threshold = [(50,[0.001, 0.002, 0.003]),
                      (100,[0.005, 0.006, 0.007]),
                      (250,[0.01, 0.012, 0.015]),
                      (500,[0.012, 0.015, 0.02]),
                      (750,[0.015, 0.02, 0.025]),
                      (1000,[0.02, 0.025, 0.03])]
    for tuple in rank_threshold:
        rank = tuple[0]
        for threshold in tuple[1]:
            params = RESCAL_BASE_CONFIG + ['--outputfolder', OUTPUT_FOLDER + '/results/rank'] + \
                     ['--rank', str(rank), '--threshold', str(threshold)]  + \
                     ['--tensorfolder', OUTPUT_FOLDER + '/tensor']
            luigi.run(params)

# evaluate the influence of stopwords on the algorithms. This test executes the preprocessing without filtering out
#  any stopwords (here in this case the effect might not be that big since only the subject line of emails is used as
#  token input)
def no_stopwords():
    params = ['AllEvaluation'] + BASE_CONFIG + RESCAL_DEFAULT_PARAMS + RESCAL2_DEFAULT_PARAMS + \
             COSINE_DEFAULT_PARAMS + ['--outputfolder', OUTPUT_FOLDER + '/results/no_stopwords'] + \
             ['--gateapp', '../../src/main/resources/gate_no_stopwords/application.xgapp']  + \
             ['--tensorfolder', OUTPUT_FOLDER + '/tensor_no_stopwords']
    luigi.run(params)

# evaluate the transitive option of the cosine distance algorithms.
# That means taking connection information into account
def cosinetrans_eval():
    COSINETRANNS_PARAMS = ['--costhreshold', '0.3', '--costransthreshold', '0.4', '--wcosthreshold', '0.3',
                           '--wcostransthreshold', '0.4']
    params = ['CosineEvaluation'] + BASE_CONFIG + COSINETRANNS_PARAMS + ['--outputfolder', OUTPUT_FOLDER + '/results/cosinetrans'] + \
             ['--tensorfolder', OUTPUT_FOLDER + '/tensor']
    luigi.run(params)

# evaluate the influence of stemming on the algorithms
def stemming_eval():
    params = ['AllEvaluation'] + BASE_CONFIG + RESCAL_DEFAULT_PARAMS + RESCAL2_DEFAULT_PARAMS + \
             COSINE_DEFAULT_PARAMS + ['--stemming'] + ['--outputfolder', OUTPUT_FOLDER + '/results/stemming'] +  \
             ['--tensorfolder', OUTPUT_FOLDER + '/tensor_stem']
    luigi.run(params)

# evaluate the effect of adding the content slice (computed by GATE, only take Noun-phrases, see gate app for details)
# to the RESCAL evaluation
def content_slice_eval():
    params = RESCAL_BASE_CONFIG + RESCAL_DEFAULT_PARAMS + ['--content', '--additionalslices', 'subject.mtx content.mtx'] + \
             ['--outputfolder', OUTPUT_FOLDER + '/results/content'] + ['--tensorfolder', OUTPUT_FOLDER + '/tensor_content']
    luigi.run(params)

# evaluate the effect of adding the category slice to the RESCAL evaluation
def category_slice_eval():
    params = ['CategoryEvaluation'] + BASE_CONFIG + ['--allneeds', TESTSET_FOLDER + '/allneeds.txt'] + \
             ['--outputfolder', OUTPUT_FOLDER + '/results/category'] + \
             ['--tensorfolder', OUTPUT_FOLDER + '/tensor_category']
    luigi.run(params + ['--rank',  '500', '--threshold', '0.02'])
    luigi.run(params + ['--rank',  '500', '--threshold', '0.03'])
    luigi.run(params + ['--rank',  '500', '--threshold', '0.04'])

# evaluate the effect of adding the needtype slice to the RESCAL evaluation
def needtype_slice_eval():
    params = RESCAL_BASE_CONFIG + RESCAL_DEFAULT_PARAMS + ['--needtypeslice'] + \
             ['--outputfolder', OUTPUT_FOLDER + '/results/needtype'] + ['--tensorfolder', OUTPUT_FOLDER + '/tensor']
    luigi.run(params)

# evaluate the effect of masking random connections instead of all connections of test needs
def maskrandom_eval():
    params = RESCAL_BASE_CONFIG + ['--outputfolder', OUTPUT_FOLDER + '/results/maskrandom'] + \
             ['--maskrandom'] + ['--tensorfolder', OUTPUT_FOLDER + '/tensor']
    luigi.run(params + ['--rank',  '500', '--threshold', '0.1'])
    luigi.run(params + ['--rank',  '500', '--threshold', '0.2'])
    luigi.run(params + ['--rank',  '500', '--threshold', '0.3'])

# evaluate the influence of the number of input connections (chosen randomly) to learn from on the RESCAL algorithm
def connections_eval():
    connection_count = [0, 1, 2, 5, 10]
    for con in connection_count:
        params = RESCAL_BASE_CONFIG + RESCAL_DEFAULT_PARAMS + RESCAL2_DEFAULT_PARAMS + \
                 ['--maxconnections', str(con)] + ['--outputfolder', OUTPUT_FOLDER + '/results/connections'] + \
                 ['--tensorfolder', OUTPUT_FOLDER + '/tensor']
        luigi.run(params)


if __name__ == '__main__':

    default_all_eval()
    no_stopwords()
    category_slice_eval()
    rank_eval()
    cosinetrans_eval()
    stemming_eval()
    needtype_slice_eval()
    maskrandom_eval()
    connections_eval()
    content_slice_eval()

