wonpreprocessing
================
Preprocessing of mail input for won matching using GATE. The MailProcessing
Java program calls a Gate application (in resources/gate) to annotate mail content.
Needs are created from WANT and OFFER mails and connections between them can
be specified in a connections file.

Afterwards a 3-way-tensor object is created as input for the rescal
algorithm (in python) that can be used to predict further connections between needs.

The input and output of RESCAL can be further evaluated with an R script.


How to run:
============

Preprocessing:
* install Gate
* set Gate home variable (-Dgate.home=<to gate base folder>)
* setup input folder with mail files
* setup output folder (including a subfolder "rescal" with "connections.txt" file)
* execute "src/main/MailProcessing.java" with following parameters:
VM options: -Dgate.home=path_to_gate -Xmx3G
program args: input_mail_folder output_mail_folder


RESCAL:
* install python (2.7 or 3.4) with scipy and numpy packages (e.g. Anaconda : http://continuum.io/downloads)
* install https://github.com/mnick/scikit-tensor
* install https://github.com/mnick/rescal.py
* execute "python-processing/src/main/python/evaluate_link_prediction.py" for an evaluation of several link
prediction algorithms, including RESCAL
* execute "python-processing/src/main/python/generate_rescal_output.py" for generation of rescal example output for preprocessed data

R:
* After execution of "python-processing/src/main/python/generate_rescal_output.py" this output can be
visualized/analyzed with R using "R-evaluation/Rgraphdraw.R" script

FEATURE_EXTRACTION:
* needs python (tested on 3.4), numpy, scipy, scikit-learn and nltk
* needs following nltk dictionaries and corpora: wordnet, maxent_treebank_pos_tagger, punkt
    - donwload by running: "import nltk; nltk.download()" from console, which runs a downloader
* execute "python-processing/src/main/python/feature_extraction.py" for printing the relevant keywords found in documents
* Soon will be able to enhance rescal tensor with new data slice containing extracted features


