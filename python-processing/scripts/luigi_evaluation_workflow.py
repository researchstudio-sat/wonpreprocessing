__author__ = 'hfriedrich'

import os
import luigi
import subprocess


# Starting task in pipeline: Normalize the file names in the input mail
# folder for later use.
class NormalizeFileNames(luigi.Task):

    inputfolder = luigi.Parameter()
    outputfolder = luigi.Parameter()

    def output(self):
        return luigi.LocalTarget(self.outputfolder)

    def run(self):
        os.system("normalize_file_names.py " + self.inputfolder + " " + self.outputfolder)


# After the file names are normalized, this task executes the Java/Gate preprocessing
# on the mail folder and create the basic tensor.
class CreateTensor(luigi.Task):

    gatehome = luigi.Parameter()
    jarfile = luigi.Parameter(default="../../target/wonpreprocessing-1.0-SNAPSHOT-jar-with-dependencies.jar")
    inputfolder = luigi.Parameter()
    tensorfolder = luigi.Parameter()
    connections = luigi.Parameter()
    gateapp = luigi.Parameter(default="../../src/main/resources/gate/application.xgapp")
    stemming = luigi.BooleanParameter(default=False)
    content = luigi.BooleanParameter(default=False)

    def requires(self):
        return [NormalizeFileNames(self.inputfolder, self.inputfolder + "/normalized")]

    def output(self):
        return luigi.LocalTarget(self.tensorfolder), \
               luigi.LocalTarget(self.tensorfolder + "/headers.txt"), \
               luigi.LocalTarget(self.tensorfolder + "/connection.mtx"), \
               luigi.LocalTarget(self.tensorfolder + "/needtype.mtx"), \
               luigi.LocalTarget(self.tensorfolder + "/subject.mtx"), \
               luigi.LocalTarget(self.tensorfolder + "/content.mtx")

    def run(self):
        java_call = "java -Xmx3G -Dgate.home=" + self.gatehome
        java_call += " -jar " + self.jarfile
        params = " -gateapp " + self.gateapp
        params += " -input " + self.input()[0].path
        params += " -output " + self.tensorfolder
        params += " -connections " + self.connections
        if (self.stemming):
            params += " -stemming"
        if (self.content):
            params += " -content"
        print subprocess.check_output(java_call + params)


# Optional task to create a category slice which maps every need to one or
# more categories.
class CreateCategorySlice(CreateTensor):

    allneeds = luigi.Parameter()

    def requires(self):
        return [CreateTensor(self.gatehome, self.jarfile,
                             self.inputfolder, self.tensorfolder,
                             self.connections, self.gateapp,
                             self.stemming, self.content)]

    def output(self):
        return luigi.LocalTarget(self.tensorfolder + "/category.mtx")

    def getParams(self):
       return self.input()[0][0].path + " " + self.allneeds

    def run(self):
        os.system("create_category_slice.py " + self.getParams())


# Use this task as a base class for different evaluation task variants
class AbstractEvaluation(CreateTensor):

    outputfolder = luigi.Parameter(default=None)
    additionalslices = luigi.Parameter(default="subject.mtx")
    maxconnections = luigi.IntParameter(default=1000)
    maskrandom = luigi.BooleanParameter(default=False)
    fbeta = luigi.FloatParameter(default=0.5)
    numneeds = luigi.IntParameter(default=10000)
    statistics = luigi.BooleanParameter(default=False)

    def requires(self):
        return [CreateTensor(self.gatehome, self.jarfile,
                             self.inputfolder, self.tensorfolder,
                             self.connections, self.gateapp,
                             self.stemming, self.content)]

    def getEvaluationFolder(self):
        return self.input()[0][0].path

    def output(self):
        return [luigi.LocalTarget(is_tmp=True)]

    def getParams(self):
        params = " -inputfolder " + self.getEvaluationFolder()
        params += " -additional_slices " + self.additionalslices
        params += " -maxconnections " + str(self.maxconnections)
        params += " -fbeta " + str(self.fbeta)
        params += " -numneeds " + str(self.numneeds)
        if (self.maskrandom):
            params += " -maskrandom "
        if (self.statistics):
            params += " -statistics "
        if (self.outputfolder):
            params += " -outputfolder " + self.outputfolder
        return params

    def run(self):
        os.system("evaluate_link_prediction.py " + self.getParams())


# Execute the evaluation for the RESCAL (optionally including RESCAL similarity) algorithm
class RESCALEvaluation(AbstractEvaluation):

    rank = luigi.IntParameter(default=0)
    threshold = luigi.FloatParameter(default=0.0)
    needtypeslice = luigi.BooleanParameter(default=False)
    rank2 = luigi.IntParameter(default=0)
    threshold2 = luigi.FloatParameter(default=0.0)

    def getParams(self):
        params = super(RESCALEvaluation, self).getParams()
        params += " -rescal " + str(self.rank) + " " + \
              str(self.threshold) + " " + str(self.needtypeslice)
        if (self.rank2 != 0):
            params += " -rescalsim " + str(self.rank2) + " " + \
                      str(self.threshold2) + " " + str(self.needtypeslice)
        return params

    def run(self):
        os.system("evaluate_link_prediction.py " + self.getParams())


# Execute the evaluation for the all algorithms
class AllEvaluation(RESCALEvaluation):

    rank2 = luigi.IntParameter()
    threshold2 = luigi.FloatParameter()
    costhreshold = luigi.FloatParameter()
    costransthreshold = luigi.FloatParameter()
    wcosthreshold = luigi.FloatParameter()
    wcostransthreshold = luigi.FloatParameter()

    def getParams(self):
        params = super(AllEvaluation, self).getParams()
        params +=" -cosine " + str(self.costhreshold) + " " + str(self.costransthreshold)
        params +=" -cosine_weighted " + str(self.wcosthreshold) + " " + str(self.wcostransthreshold)
        return params

    def run(self):
        os.system("evaluate_link_prediction.py " + self.getParams())

# Execute the evaluation for the cosine algorithm
class CosineEvaluation(AbstractEvaluation):

    costhreshold = luigi.FloatParameter()
    costransthreshold = luigi.FloatParameter()
    wcosthreshold = luigi.FloatParameter()
    wcostransthreshold = luigi.FloatParameter()

    def getParams(self):
        params = super(CosineEvaluation, self).getParams()
        params +=" -cosine " + str(self.costhreshold) + " " + str(self.costransthreshold)
        params +=" -cosine_weighted " + str(self.wcosthreshold) + " " + str(self.wcostransthreshold)
        return params

    def run(self):
        os.system("evaluate_link_prediction.py " + self.getParams())

# Execute the evaluation for the RESCAL (optionally including RESCAL similarity)
# algorithm, including an additional category slice
class CategoryEvaluation(RESCALEvaluation):

    allneeds = luigi.Parameter()

    def getParams(self):
        self.additionalslices = "subject.mtx category.mtx"
        params = super(CategoryEvaluation, self).getParams()
        return params

    def requires(self):
        return [CreateTensor(self.gatehome, self.jarfile,
                             self.inputfolder, self.tensorfolder,
                             self.connections, self.gateapp,
                             self.stemming, self.content),
                CreateCategorySlice(self.gatehome, self.jarfile,
                                    self.inputfolder, self.tensorfolder,
                                    self.connections, self.gateapp,
                                    self.stemming, self.content,
                                    self.allneeds)]

    def run(self):
        os.system("evaluate_link_prediction.py " + self.getParams())


