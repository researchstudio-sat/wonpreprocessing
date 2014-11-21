wonpreprocessing
================
The projects implements preprocessing of mail input and data creation for won matching evaluation.
The whole process is implemented in python using luigi (https://github.com/spotify/luigi).
Different tasks are called in this process (e.g. Java-based tasks, python-based tasks)

The 'MailProcessing' Java program calls a Gate application (in src/main/resources/) to annotate mail content.
Needs are created from WANT and OFFER mails and connections between them can be specified in a connections file.

Afterwards a 3-way-tensor object is created as input to evaluate different algorithms (e.g. RESCAL)
in python ('evaluate_link_prediction.py') that can be used to predict further connections between needs.


What to install:
================
* install Gate
* install python (2.7 or 3.4) with scipy and numpy packages (e.g. Anaconda : http://continuum.io/downloads)
* install python luigi package (https://github.com/spotify/luigi)
* install https://github.com/mnick/scikit-tensor
* install https://github.com/mnick/rescal.py


How to run:
============
* a test data set (e.g. 'testdataset_20141112.zip') is needed to run the evaluation on
* extract the test data set to a test data set folder
* execute maven build (package) of this project to build the 'wonpreprocessing-1.0-SNAPSHOT-jar-with-dependencies.jar'
* the whole process can be executed by starting the script 'luigi_evaluation.py' with its parameters
* check the script for details




R:
* After execution of "python-processing/src/main/python/generate_rescal_output.py" this output can be
visualized/analyzed with R using "R-evaluation/Rgraphdraw.R" script

FEATURE_EXTRACTION:
* needs python (tested on 3.4), numpy, scipy, scikit-learn and nltk
* needs following nltk dictionaries and corpora: wordnet, maxent_treebank_pos_tagger, punkt
    - donwload by running: "import nltk; nltk.download()" from console, which runs a downloader
* execute "python-processing/src/main/python/feature_extraction.py" for printing the relevant keywords found in documents
* Soon will be able to enhance rescal tensor with new data slice containing extracted features


