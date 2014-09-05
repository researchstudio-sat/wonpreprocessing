wonpreprocessing
================
Preprocessing of mail input for won matching using GATE. The MailProcessing Java
program calls a Gate application (in resources/gate) to annotate mail content.
Needs are created from WANT and OFFER mails and connections between them can
be specified in a connections file.
Afterwards a 3-way-tensor object is created as input for the rescal
algorithm (in python) that can be used to predict further connections between needs.


How to run:
============

Preprocessing:
* install Gate
* set Gate home variable (-Dgate.home=<to gate base folder>)
* setup input folder with mail files
* setup output folder (including a subfolder "rescal" with "connections.txt" file)
* execute "MailProcessing.java" with following parameters:
VM options: -Dgate.home=<pathtogate> -Xmx3G
program args: <input mail folder> <output mail folder>


RESCAL:
* install python (2.7 or 3.4) with scipy and numpy packages (e.g. Anaconda : http://continuum.io/downloads)
* install https://github.com/mnick/scikit-tensor
* install https://github.com/mnick/rescal.py
* execute "evaluate_rescal.py" for an evaluation of the preprocessed that with rescal
* execute "generate_rescal_output.py" for generation of rescal example output for preprocessed data
* After execution of "generate_rescal_output.py" this output can be visualized/analyzed with R using "graphdraw.R"
script



