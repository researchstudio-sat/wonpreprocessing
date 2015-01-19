__author__ = 'hfriedrich'

import os
import luigi
import subprocess


def run_python(python_path, module_path, *args):
    """Helper for running python scripts.

    :param python_path: Path to python interpreter.
    :param module_path: Path to python module.
    :param args: Arguments for python module.
    :return: None
    """
    command = [python_path, module_path]
    command.extend(args)
    os.system(" ".join(command))


# Starting task in pipeline: Normalize the file names in the input mail
# folder for later use.
class NormalizeFileNames(luigi.Task):

    inputfolder = luigi.Parameter()
    outputfolder = luigi.Parameter()
    python = luigi.Parameter(default='python')

    def output(self):
        return luigi.LocalTarget(self.outputfolder)

    def run(self):
        os.system("" + self.python + " normalize_file_names.py " + self.inputfolder + " " + self.outputfolder)


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
    java = luigi.Parameter(default='java')
    python = luigi.Parameter(default='python')

    def requires(self):
        return [NormalizeFileNames(self.inputfolder, self.inputfolder + "/normalized", self.python)]

    def output(self):
        return luigi.LocalTarget(self.tensorfolder), \
               luigi.LocalTarget(self.tensorfolder + "/headers.txt"), \
               luigi.LocalTarget(self.tensorfolder + "/connection.mtx"), \
               luigi.LocalTarget(self.tensorfolder + "/needtype.mtx"), \
               luigi.LocalTarget(self.tensorfolder + "/subject.mtx"), \
               luigi.LocalTarget(self.tensorfolder + "/content.mtx")

    def run(self):
        java_call = [self.java, '-Xmx3G', '-Dgate.home=' + self.gatehome,
                     '-jar', self.jarfile]
        params = ['-gateapp', self.gateapp,
                  '-input', self.input()[0].path,
                  '-output', self.tensorfolder,
                  '-connections', self.connections]
        if self.stemming:
            params.append('-stemming')
        if self.content:
            params.append('-content')
        print subprocess.check_output(java_call + params)


# Optional task to create a category slice which maps every need to one or
# more categories.
class CreateCategorySlice(CreateTensor):

    allneeds = luigi.Parameter()

    def requires(self):
        return [CreateTensor(self.gatehome, self.jarfile,
                             self.inputfolder, self.tensorfolder,
                             self.connections, self.gateapp,
                             self.stemming, self.content,
                             self.java, self.python)]

    def output(self):
        return luigi.LocalTarget(self.tensorfolder + "/category.mtx")

    def getParams(self):
       return self.input()[0][0].path + " " + self.allneeds

    def run(self):
        os.system("" + self.python + "create_category_slice.py " + self.getParams())


class CreateKeywordSlice(CreateTensor):
    """Optional task to create a keywords slice"""

    # ngram_range = luigi.Parameter()

    def requires(self):
        return [CreateTensor(self.gatehome, self.jarfile,
                             self.inputfolder, self.tensorfolder,
                             self.connections, self.gateapp,
                             self.stemming, self.content,
                             self.java, self.python)]

    def output(self):
        return luigi.LocalTarget(self.tensorfolder + '/keyword.mtx')

    def run(self):
        run_python(self.python, 'add_keyword_slice.py', self.inputfolder,
                   self.tensorfolder)


# Use this task as a base class for different evaluation task variants
class BaseEvaluation(CreateTensor):

    outputfolder = luigi.Parameter(default=None)
    additionalslices = luigi.Parameter(default="subject.mtx")
    maxconnections = luigi.IntParameter(default=1000)
    maskrandom = luigi.BooleanParameter(default=False)
    fbeta = luigi.FloatParameter(default=0.5)
    numneeds = luigi.IntParameter(default=10000)
    statistics = luigi.BooleanParameter(default=True)
    maxhubsize = luigi.IntParameter(default=10000)

    def requires(self):
        return [CreateTensor(self.gatehome, self.jarfile,
                             self.inputfolder, self.tensorfolder,
                             self.connections, self.gateapp,
                             self.stemming, self.content,
                             self.java, self.python)]

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
        params += " -maxhubsize " + str(self.maxhubsize)
        if (self.maskrandom):
            params += " -maskrandom "
        if (self.statistics):
            params += " -statistics "
        if (self.outputfolder):
            params += " -outputfolder " + self.outputfolder
        return params

    def run(self):
        os.system("" + self.python + " evaluate_link_prediction.py " + self.getParams())


# Execute the evaluation for the RESCAL (optionally including RESCAL similarity) algorithm
class RESCALEvaluation(BaseEvaluation):

    rank = luigi.IntParameter(default=0)
    threshold = luigi.FloatParameter(default=0.0)
    needtypeslice = luigi.BooleanParameter(default=False)
    transitive = luigi.BooleanParameter(default=False)
    init = luigi.Parameter(default='nvecs')
    conv = luigi.FloatParameter(default=1e-4)
    lambdaA = luigi.FloatParameter(default=0.0)
    lambdaR = luigi.FloatParameter(default=0.0)
    lambdaV = luigi.FloatParameter(default=0.0)
    rank2 = luigi.IntParameter(default=0)
    threshold2 = luigi.FloatParameter(default=0.0)
    connectionslice2 = luigi.BooleanParameter(default=False)

    def getParams(self):
        params = super(RESCALEvaluation, self).getParams()
        if (self.rank != 0):
            params += " -rescal " + str(self.rank) + " " + \
                str(self.threshold) + " " + str(self.needtypeslice) + " " + str(self.transitive) + " " + self.init + \
                      " " + str(self.conv) + " " + str(self.lambdaA) + " " + str(self.lambdaR) + " " + str(self.lambdaV)
        if (self.rank2 != 0):
            params += " -rescalsim " + str(self.rank2) + " " + \
                str(self.threshold2) + " " + str(self.needtypeslice) + " " + str(self.connectionslice2)
        return params


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

# Execute the evaluation for the cosine algorithm
class CosineEvaluation(BaseEvaluation):

    costhreshold = luigi.FloatParameter()
    costransthreshold = luigi.FloatParameter()
    wcosthreshold = luigi.FloatParameter()
    wcostransthreshold = luigi.FloatParameter()

    def getParams(self):
        params = super(CosineEvaluation, self).getParams()
        params +=" -cosine " + str(self.costhreshold) + " " + str(self.costransthreshold)
        params +=" -cosine_weighted " + str(self.wcosthreshold) + " " + str(self.wcostransthreshold)
        return params

# Execute the evaluation for the combined version of cosine and rescal algorithm
class CombineCosineRescalEvaluation(BaseEvaluation):

    rank = luigi.IntParameter()
    rescalthreshold = luigi.FloatParameter()
    cosinethreshold = luigi.FloatParameter()
    needtypeslice = luigi.BooleanParameter(default=False)

    def getParams(self):
        params = super(CombineCosineRescalEvaluation, self).getParams()
        params +=" -cosine_rescal " + str(self.rank) + " " + str(self.rescalthreshold) + \
                 " " + str(self.cosinethreshold) + " " + str(self.needtypeslice)
        return params

# Execute the evaluation for the prediction intersection between cosine and rescal algorithm
class IntersectionEvaluation(BaseEvaluation):

    rank = luigi.IntParameter()
    rescalthreshold = luigi.FloatParameter()
    cosinethreshold = luigi.FloatParameter()
    needtypeslice = luigi.BooleanParameter(default=False)

    def getParams(self):
        params = super(IntersectionEvaluation, self).getParams()
        params +=" -intersection " + str(self.rank) + " " + str(self.rescalthreshold) + \
                 " " + str(self.cosinethreshold) + " " + str(self.needtypeslice)
        return params

# Execute the evaluation for the RESCAL (optionally including RESCAL similarity)
# algorithm, including an additional category slice
class CategoryEvaluation(RESCALEvaluation):

    allneeds = luigi.Parameter()

    def getParams(self):
        self.additionalslices = "subject.mtx content.mtx category.mtx"
        params = super(CategoryEvaluation, self).getParams()
        return params

    def requires(self):
        return [CreateTensor(self.gatehome, self.jarfile,
                             self.inputfolder, self.tensorfolder,
                             self.connections, self.gateapp,
                             self.stemming, self.content,
                             self.java, self.python),
                CreateCategorySlice(self.gatehome, self.jarfile,
                                    self.inputfolder, self.tensorfolder,
                                    self.connections, self.gateapp,
                                    self.stemming, self.content,
                                    self.java, self.python,
                                    self.allneeds)]


class KeywordEvaluation(RESCALEvaluation):

    def getParams(self):
        self.additionalslices = "keyword.mtx"
        params = super(KeywordEvaluation, self).getParams()
        return params

    def requires(self):
        return [
            CreateTensor(self.gatehome, self.jarfile,
                         self.inputfolder, self.tensorfolder,
                         self.connections, self.gateapp,
                         self.stemming, self.content,
                         self.java, self.python),
            CreateKeywordSlice(self.gatehome, self.jarfile,
                               self.inputfolder, self.tensorfolder,
                               self.connections, self.gateapp,
                               self.stemming, self.content,
                               self.java, self.python)
        ]

