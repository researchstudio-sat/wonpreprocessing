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
        os.system("normalize_file_names.py " + self.inputfolder + " " + self.inputfolder + "/normalized")


# After the file names are normalized, this task executes the Java/Gate preprocessing
# on the mail folder and create the basic tensor.
class CreateTensor(luigi.Task):

    gatehome = luigi.Parameter()
    jarfile = luigi.Parameter(default="../../target/wonpreprocessing-1.0-SNAPSHOT-jar-with-dependencies.jar")
    inputfolder = luigi.Parameter()
    outputfolder = luigi.Parameter()
    connections = luigi.Parameter()
    gateapp = luigi.Parameter(default="../../src/main/resources/gate/application.xgapp")
    stemming = luigi.BooleanParameter(default=False)
    content = luigi.BooleanParameter(default=False)

    def getOutputFolder(self):
        out = self.outputfolder + "/results"
        if (self.stemming):
            out += "_stem"
        if (self.content):
            out += "_content"
        return out

    def requires(self):
        return [NormalizeFileNames(self.inputfolder, self.inputfolder + "/normalized")]

    def output(self):
        return luigi.LocalTarget(self.getOutputFolder()), \
               luigi.LocalTarget(self.getOutputFolder() + "/headers.txt"), \
               luigi.LocalTarget(self.getOutputFolder() + "/connection.mtx"), \
               luigi.LocalTarget(self.getOutputFolder() + "/needtype.mtx"), \
               luigi.LocalTarget(self.getOutputFolder() + "/subject.mtx"), \
               luigi.LocalTarget(self.getOutputFolder() + "/content.mtx")

    def run(self):
        java_call = "java -Xmx3G -Dgate.home=" + self.gatehome
        java_call += " -jar " + self.jarfile
        params = " -gateapp " + self.gateapp
        params += " -input " + self.input()[0].path
        params += " -output " + self.getOutputFolder()
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
                             self.inputfolder, self.outputfolder,
                             self.connections, self.gateapp,
                             self.stemming, self.content)]

    def output(self):
        return luigi.LocalTarget(self.getOutputFolder() + "/category.mtx")

    def getParams(self):
       return self.input()[0][0].path + " " + self.allneeds

    def run(self):
        os.system("create_category_slice.py " + self.getParams())


# Use this task as a base class for different evaluation task variants
class AbstractEvaluation(CreateTensor):

    additional_slices = luigi.Parameter(default="subject.mtx")
    maxconnections = luigi.IntParameter(default=1000)
    maskrandom = luigi.BooleanParameter(default=False)
    fbeta = luigi.FloatParameter(default=0.5)
    statistics = luigi.BooleanParameter(default=False)

    def requires(self):
        return [CreateTensor(self.gatehome, self.jarfile,
                             self.inputfolder, self.outputfolder,
                             self.connections, self.gateapp,
                             self.stemming, self.content)]

    def getEvaluationFolder(self):
        return self.input()[0][0].path

    def output(self):
        return [luigi.LocalTarget(is_tmp=True)]

    def getParams(self):
        params = " -folder " + self.getEvaluationFolder()
        params += " -additional_slices " + self.additional_slices
        params += " -maxconnections " + str(self.maxconnections)
        params += " -fbeta " + str(self.fbeta)
        if (self.maskrandom):
            params += " -maskrandom "
        if (self.statistics):
            params += " -statistics "
        return params

    def run(self):
        os.system("evaluate_link_prediction.py " + self.getParams())


# Execute the evaluation for the RESCAL (optionally including RESCAL similarity) algorithm
class RESCALEvaluation(AbstractEvaluation):

    rank = luigi.IntParameter()
    threshold = luigi.FloatParameter()
    needtypeSlice = luigi.BooleanParameter(default=False)
    rank2 = luigi.IntParameter(default=0)
    threshold2 = luigi.FloatParameter(default=0.0)

    def getParams(self):
        params = super(RESCALEvaluation, self).getParams()
        params += " -rescal " + str(self.rank) + " " + \
              str(self.threshold) + " " + str(self.needtypeSlice)
        if (self.rank2 != 0):
            params += " -rescalsim " + str(self.rank2) + " " + \
                      str(self.threshold2) + " " + str(self.needtypeSlice)
        return params

    def run(self):
        os.system("evaluate_link_prediction.py " + self.getParams())


# Execute the evaluation for the all algorithms
class AllEvaluation(RESCALEvaluation):

    rank2 = luigi.IntParameter()
    threshold2 = luigi.FloatParameter()
    cosine_threshold = luigi.FloatParameter()
    cosine_transthreshold = luigi.FloatParameter()
    wcosine_threshold = luigi.FloatParameter()
    wcosine_transthreshold = luigi.FloatParameter()

    def getParams(self):
        params = super(AllEvaluation, self).getParams()
        params +=" -cosine " + str(self.cosine_threshold) + " " + str(self.cosine_transthreshold)
        params +=" -cosine_weighted " + str(self.wcosine_threshold) + " " + str(self.wcosine_transthreshold)
        return params

    def run(self):
        os.system("evaluate_link_prediction.py " + self.getParams())

# Execute the evaluation for the RESCAL (optionally including RESCAL similarity)
# algorithm, including an additional category slice
class CategoryEvaluation(RESCALEvaluation):

    allneeds = luigi.Parameter()

    def getParams(self):
        self.additional_slices = "subject.mtx category.mtx"
        params = super(CategoryEvaluation, self).getParams()
        return params

    def requires(self):
        return [CreateTensor(self.gatehome, self.jarfile,
                             self.inputfolder, self.outputfolder,
                             self.connections, self.gateapp,
                             self.stemming, self.content),
                CreateCategorySlice(self.gatehome, self.jarfile,
                                    self.inputfolder, self.outputfolder,
                                    self.connections, self.gateapp,
                                    self.stemming, self.content,
                                    self.allneeds)]

    def run(self):
        os.system("evaluate_link_prediction.py " + self.getParams())

if __name__ == '__main__':
    # CategoryEvaluation --lock-pid-dir C:\dev\temp\luigi --local-scheduler --gatehome C:\dev\GATE_Developer_8.0 --inputfolder C:/dev/temp/testcorpus/complete/std2 --outputfolder C:/dev/temp/testcorpus/complete/stdout --connections C:/dev/temp/testcorpus/complete/out/rescal/connections.txt --allneeds C:/dev/temp/testcorpus/complete/out/rescal/allneeds3.txt --stemming --rank 200 --threshold 0.01
    luigi.run()

