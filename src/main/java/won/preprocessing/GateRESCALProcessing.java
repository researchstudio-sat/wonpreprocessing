/*
 * Copyright 2012  Research Studios Austria Forschungsges.m.b.H.
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 */

package won.preprocessing;

import gate.*;
import gate.creole.ExecutionException;
import gate.creole.ResourceInstantiationException;
import gate.util.GateException;
import gate.util.persistence.PersistenceManager;
import org.apache.commons.io.FilenameUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.net.MalformedURLException;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;

/**
 * User: hfriedrich
 * Date: 01.07.2014
 *
 * Create data (binary tensor) for the RESCAL algorithm from a corpus of text files that are processed by Gate.
 */
public class GateRESCALProcessing
{
  private static final Logger logger = LoggerFactory.getLogger(GateRESCALProcessing.class);

  public enum AnnotationType {
    TOPIC("TopicToken", "string"),
    DESCRIPTION("DescriptionToken", "string"),
    CLASSIFICATION("NeedClassificationToken", "kind");

    private String tokenName;
    private String featureName;

    private AnnotationType(String token, String feature) {
      tokenName = token;
      featureName = feature;
    }

    public String getTokenName() {
      return tokenName;
    }
    public String getFeatureName() {
      return featureName;
    }

    public static AnnotationType getTypeByToken(String token) {
      for (AnnotationType type: AnnotationType.values()) {
        if (type.getTokenName().equals(token)) {
          return type;
        }
      }
      return null;
    }
  }

  private String baseFolder;
  private WonMatchingData matchingData;
  private CorpusController gateApplication;

  public GateRESCALProcessing(String gateAppPath, String baseFolder) throws GateException, IOException {
    matchingData = new WonMatchingData();
    this.baseFolder = baseFolder;

    // init Gate
    logger.info("Initialising Gate");
    Gate.init();

    // load Gate application
    logger.info("Loading Gate application: {}", gateAppPath);
    gateApplication = (CorpusController)
      PersistenceManager.loadObjectFromFile(new File(gateAppPath));
  }

  /**
   * After the mails have been preprocessed the Gate processing and tensor creation is executed by the gate
   * application.
   *
   * @param corpusFolder corpus document input folder
   * @throws gate.util.GateException
   * @throws MalformedURLException
   */
  public void processFilesWithGate(String corpusFolder)
    throws IOException, ResourceInstantiationException, ExecutionException {

    int maxFilesPerCorpus = 1000;
    int processedFiles = 0;

    logger.info("Gate processing and tensor creation of files from folder: {}", corpusFolder);
    File folder = new File(corpusFolder);

    for (int bulk = 0; bulk < folder.listFiles().length; bulk += maxFilesPerCorpus) {
      Corpus corpus = Factory.newCorpus("Transient Gate Corpus");
      for (int i = bulk; i < maxFilesPerCorpus + bulk && i < folder.listFiles().length; i++) {
        File file = folder.listFiles()[i];
        if (!file.isDirectory()) {
          corpus.add(Factory.newDocument(file.toURI().toURL()));
        }
      }
      gateApplication.setCorpus(corpus);
      gateApplication.execute();
      addDataFromProcessedCorpus(corpus);
      processedFiles += corpus.size();
      logger.info("{} files processed ...", processedFiles);
    }
  }

  private static void saveXMLDocumentAnnotations(Corpus corpus, String folder) throws IOException {

    logger.info("Saving XML gate annotation files to folder: {}", folder);
    File outFolder = new File(folder);
    outFolder.mkdirs();
    Iterator documentIterator = corpus.iterator();
    while(documentIterator.hasNext()) {
      Document currDoc = (Document) documentIterator.next();
      String xmlDocument = currDoc.toXml();
      String fileName = java.net.URLDecoder.decode(FilenameUtils.getBaseName(currDoc.getSourceUrl().getFile()),
                                                   "UTF-8");
      String path = new String(folder + "/" + fileName + ".xml");
      logger.debug("Saving XML gate annotation file: {}", path);
      FileWriter writer = new FileWriter(path);
      writer.write(xmlDocument);
      writer.close();
    }
  }

  /**
   * Add a Gate-processed corpus to the class to generate output of it later by {@link #createRescalData(String)}.
   * The documents in the corpus must have been annotated correctly by Gate (see annotation definition constants in
   * this class).
   *
   * @param corpus
   */
  private void addDataFromProcessedCorpus(Corpus corpus) throws UnsupportedEncodingException {
    Iterator documentIterator = corpus.iterator();
    while (documentIterator.hasNext()) {
      Document currDoc = (Document) documentIterator.next();
      String needId = createNeedId(currDoc);

      for (Annotation annotation : currDoc.getAnnotations()) {

        String attrValue = null;
        AnnotationType type = AnnotationType.getTypeByToken(annotation.getType());
        if (type == null) {
          continue;
        }

        switch (type) {

          case TOPIC:
            attrValue = getFeatureValueFromAnnotation(annotation, type.getFeatureName());
            matchingData.addNeedAttribute(needId, attrValue, WonMatchingData.AttributeType.TOPIC);
            break;

          case DESCRIPTION:
            attrValue = getFeatureValueFromAnnotation(annotation, type.getFeatureName());
            //matchingData.addNeedAttribute(needId, attrValue, WonMatchingData.AttributeType.DESCRIPTION);
            break;

          case CLASSIFICATION:
            attrValue = getFeatureValueFromAnnotation(annotation, type.getFeatureName());
            WonMatchingData.NeedType needType = WonMatchingData.NeedType.OFFER;
            if (!attrValue.equalsIgnoreCase(needType.name())) {
              needType = WonMatchingData.NeedType.WANT;
              if(!attrValue.equalsIgnoreCase(needType.name())) {
                logger.error("Unknown feature value '{}' found in annotation '{}'",
                             type.getFeatureName() + "=" + attrValue,
                             annotation.getType());
                break;
              }
            }
            matchingData.addNeedType(needId, needType);
            break;

          default:
            break;
        }
      }
    }
  }

  private String getFeatureValueFromAnnotation(Annotation annotation, String feature) {
    String attrValue = (String) annotation.getFeatures().get(feature);
    if (attrValue == null) {
      logger.error("Feature value '{}' not found in annotation '{}'", feature,
                   annotation.getType());
    }
    return attrValue.toLowerCase();
  }

  /**
   * Add data about Need connections. Use filenames as names for needs, one need per line. Create a need connection
   * between a Need and all following Needs until empty line in text file.
   *
   */
  public void addConnectionData(String connectionFile) throws IOException {

    logger.info("Create Need connection from input file: {}", connectionFile);
    BufferedReader reader = new BufferedReader(new FileReader(connectionFile));
    String line = "";
    List<String> needs = new LinkedList<String>();

    while ((line = reader.readLine()) != null) {
      if (line.length() == 0) {
        // add a connection between the first need and all following needs until empty line
        addConnection(needs);
        needs = new LinkedList<String>();
      } else {
        needs.add(line.trim());
      }
    }
    addConnection(needs);
  }

  private String createNeedId(Document doc) throws UnsupportedEncodingException {
    return java.net.URLDecoder.decode(FilenameUtils.getBaseName(doc.getSourceUrl().getFile()), "UTF-8");
  }

  private void addConnection(List<String> needs) throws MalformedURLException {
    for (int i = 1; i < needs.size(); i++) {
      String need1 = MailProcessing.cleanFileName(needs.get(0));
      String need2 = MailProcessing.cleanFileName(needs.get(i));
      if (!matchingData.getNeeds().contains(need1) || !matchingData.getNeeds().contains(need2)) {
        logger.warn("add connection between new needs: \n{} \n{}", need1, need2);
      }
      matchingData.addNeedConnection(need1, need2);
    }
  }

  /**
   * Save the data that was added using {@link #addDataFromProcessedCorpus(gate.Corpus)}.
   *
   * @param outputFolder data folder
   * @throws IOException
   */
  public void createRescalData(String outputFolder) throws IOException {
    matchingData.writeOutputFiles(outputFolder);
  }

}
