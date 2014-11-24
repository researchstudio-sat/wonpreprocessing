__author__ = 'hfriedrich'

import luigi
import argparse
import luigi_evaluation_workflow


# This is the executed experiments script for the link prediction evaluation (e.g. on the 'testdataset_20141112').
# It executes the luigi_evaluation_workflow with the parameters specified below.
#
# how to run (for installation see README.MD):
# - Extract the test data set to a test data set folder
# - execute maven build (package) of this project to build the 'wonpreprocessing-1.0-SNAPSHOT-jar-with-dependencies.jar'
# - run this python script with its parameters

RESCAL_DEFAULT_PARAMS = ['--rank',  '500', '--threshold', '0.02']
RESCAL2_DEFAULT_PARAMS = ['--rank2',  '500', '--threshold2', '0.05']
COSINE_DEFAULT_PARAMS = ['--costhreshold', '0.5', '--costransthreshold', '0.0', '--wcosthreshold', '0.5', '--wcostransthreshold', '0.0']

def output_folder_config():
    return args.testdataset + '/evaluation'

def base_config():
    return ['--lock-pid-dir', args.luigitmp, '--local-scheduler', '--gatehome', args.gatehome,
            '--inputfolder', args.testdataset + '/data', '--connections', args.testdataset + '/connections.txt']

# evaluate all algorithms in their default configuration
def default_all_eval():
    params = ['AllEvaluation'] + base_config() + RESCAL_DEFAULT_PARAMS + RESCAL2_DEFAULT_PARAMS + \
             COSINE_DEFAULT_PARAMS + ['--outputfolder', output_folder_config() + '/results/default'] + \
             ['--tensorfolder', output_folder_config() + '/tensor']
    luigi.run(params)

# evaluate the influence of the rank on the quality and performance of link prediction
def rank_eval():
    rank_threshold = [(50,[0.001, 0.002, 0.003]),
                      (100,[0.005, 0.006, 0.007]),
                      (250,[0.01, 0.012, 0.015]),
                      (500,[0.012, 0.015, 0.02]),
                      (750,[0.015, 0.02, 0.025]),
                      (1000,[0.02, 0.025, 0.03]),
                      (2000,[0.02, 0.025, 0.03])]
    for tuple in rank_threshold:
        rank = tuple[0]
        for threshold in tuple[1]:
            params = ['RESCALEvaluation'] + base_config() + \
                     ['--outputfolder', output_folder_config() + '/results/rank'] + \
                     ['--rank', str(rank), '--threshold', str(threshold)]  + \
                     ['--tensorfolder', output_folder_config() + '/tensor']
            luigi.run(params)

# evaluate the influence of stopwords on the algorithms. This test executes the preprocessing without filtering out
#  any stopwords (here in this case the effect might not be that big since only the subject line of emails is used as
#  token input)
def no_stopwords():
    params = ['AllEvaluation'] + base_config() + RESCAL_DEFAULT_PARAMS + RESCAL2_DEFAULT_PARAMS + \
             COSINE_DEFAULT_PARAMS + ['--outputfolder', output_folder_config() + '/results/no_stopwords'] + \
             ['--gateapp', '../../src/main/resources/gate_no_stopwords/application.xgapp']  + \
             ['--tensorfolder', output_folder_config() + '/tensor_no_stopwords']
    luigi.run(params)

# evaluate the transitive option of the cosine distance algorithms.
# That means taking connection information into account
def cosinetrans_eval():
    COSINETRANNS_PARAMS = ['--costhreshold', '0.2', '--costransthreshold', '0.25', '--wcosthreshold', '0.2',
                           '--wcostransthreshold', '0.25']
    params = ['CosineEvaluation'] + base_config() + COSINETRANNS_PARAMS + ['--outputfolder', output_folder_config() + '/results/cosinetrans'] + \
             ['--tensorfolder', output_folder_config() + '/tensor']
    luigi.run(params)

# evaluate the influence of stemming on the algorithms
def stemming_eval():
    params = ['AllEvaluation'] + base_config() + RESCAL_DEFAULT_PARAMS + RESCAL2_DEFAULT_PARAMS + \
             COSINE_DEFAULT_PARAMS + ['--stemming'] + ['--outputfolder', output_folder_config() + '/results/stemming'] +  \
             ['--tensorfolder', output_folder_config() + '/tensor_stem']
    luigi.run(params)

# evaluate the effect of adding the content slice (computed by GATE, only take Noun-phrases, see gate app for details)
# to the RESCAL evaluation
def content_slice_eval():
    params = ['RESCALEvaluation'] + base_config() + ['--content', '--additionalslices', 'subject.mtx content.mtx'] + \
             ['--outputfolder', output_folder_config() + '/results/content'] + \
             ['--tensorfolder', output_folder_config() + '/tensor_content']
    luigi.run(params + RESCAL_DEFAULT_PARAMS)
    luigi.run(params + ['--rank',  '500', '--threshold', '0.03'])

# evaluate the effect of adding the category slice to the RESCAL evaluation
def category_slice_eval():
    params = ['CategoryEvaluation'] + base_config() + ['--allneeds', args.testdataset + '/allneeds.txt'] + \
             ['--outputfolder', output_folder_config() + '/results/category'] + \
             ['--tensorfolder', output_folder_config() + '/tensor_category']
    luigi.run(params + ['--rank',  '500', '--threshold', '0.02'])
    luigi.run(params + ['--rank',  '500', '--threshold', '0.03'])
    luigi.run(params + ['--rank',  '500', '--threshold', '0.04'])

# evaluate the effect of adding the needtype slice to the RESCAL evaluation
def needtype_slice_eval():
    params = ['RESCALEvaluation'] + base_config() + RESCAL_DEFAULT_PARAMS + ['--needtypeslice'] + \
             ['--outputfolder', output_folder_config() + '/results/needtype'] + ['--tensorfolder', output_folder_config() + '/tensor']
    luigi.run(params)

# evaluate the effect of masking random connections instead of all connections of test needs
def maskrandom_eval():
    params = ['RESCALEvaluation'] + base_config() + ['--outputfolder', output_folder_config() + '/results/maskrandom'] + \
             ['--maskrandom'] + ['--tensorfolder', output_folder_config() + '/tensor']
    luigi.run(params + ['--rank',  '500', '--threshold', '0.1'])
    luigi.run(params + ['--rank',  '500', '--threshold', '0.2'])
    luigi.run(params + ['--rank',  '500', '--threshold', '0.3'])

# evaluate the influence of the number of input connections (chosen randomly) to learn from on the RESCAL algorithm
def connection_rescalsim_eval():
    connection_count = [0, 1, 2, 5, 10]
    for con in connection_count:
        params = ['RESCALEvaluation'] + base_config() + RESCAL2_DEFAULT_PARAMS + \
                 ['--maxconnections', str(con)] + ['--outputfolder', output_folder_config() + '/results/connections'] + \
                 ['--tensorfolder', output_folder_config() + '/tensor']
        luigi.run(params)

# evaluate the influence of the number of input connections (chosen randomly) to learn from on the RESCALSIM algorithm
def connections_rescal_eval():
    connection_threshold = [(1,[0.007, 0.01, 0.02]),
                            (2,[0.01, 0.02]),
                            (5,[0.02, 0.03]),
                            (10,[0.02, 0.03])]
    for tuple in connection_threshold:
        for threshold in tuple[1]:
            con = tuple[0]
            params = ['RESCALEvaluation'] + base_config() + ['--rank',  '500', '--threshold', str(threshold)] + \
                     ['--maxconnections', str(con)] + ['--outputfolder', output_folder_config() + '/results/connections'] + \
                     ['--tensorfolder', output_folder_config() + '/tensor']
            luigi.run(params)

def num_needs_eval():
    params = ['AllEvaluation'] + base_config() + \
             ['--outputfolder', output_folder_config() + '/results/num_needs'] + \
             ['--tensorfolder', output_folder_config() + '/tensor']

    luigi.run(params + ['--rank',  '500', '--threshold', '0.06', '--numneeds', '500'] +
              ['--rank2',  '500', '--threshold2', '0.03'] +
              ['--costhreshold', '0.6', '--costransthreshold', '0.0'] +
              ['--wcosthreshold', '0.6', '--wcostransthreshold', '0.0'])
    luigi.run(params + ['--rank',  '500', '--threshold', '0.05', '--numneeds', '1000'] +
              ['--rank2',  '500', '--threshold2', '0.04'] +
              ['--costhreshold', '0.6', '--costransthreshold', '0.0'] +
              ['--wcosthreshold', '0.6', '--wcostransthreshold', '0.0'])
    luigi.run(params + ['--rank',  '500', '--threshold', '0.04', '--numneeds', '2000'] +
              ['--rank2',  '500', '--threshold2', '0.04'] +
              ['--costhreshold', '0.5', '--costransthreshold', '0.0'] +
              ['--wcosthreshold', '0.5', '--wcostransthreshold', '0.0'])
    luigi.run(params + ['--rank',  '500', '--threshold', '0.03', '--numneeds', '3000'] +
              ['--rank2',  '500', '--threshold2', '0.05'] +
              ['--costhreshold', '0.5', '--costransthreshold', '0.0'] +
              ['--wcosthreshold', '0.5', '--wcostransthreshold', '0.0'])

if __name__ == '__main__':

    # CLI processing
    parser = argparse.ArgumentParser(description='link prediction evaluation on a test data set')
    parser.add_argument('-testdataset',
                        action="store", dest="testdataset", required=True,
                        help="test data set folder of the evaluation (structure of folder must be like "
                             "e.g. testdataset_20141112.zip")
    parser.add_argument('-gatehome',
                        action="store", dest="gatehome", required=True,
                        help="folder to gate home")
    parser.add_argument('-luigitmp',
                        action="store", dest="luigitmp", required=False,
                        help="folder to luigi tmp folder")
    args = parser.parse_args()

    # run the experiments
    default_all_eval()
    no_stopwords()
    category_slice_eval()
    stemming_eval()
    needtype_slice_eval()
    maskrandom_eval()
    content_slice_eval()
    cosinetrans_eval()
    num_needs_eval()
    rank_eval()
    connections_rescal_eval()
    connection_rescalsim_eval()


