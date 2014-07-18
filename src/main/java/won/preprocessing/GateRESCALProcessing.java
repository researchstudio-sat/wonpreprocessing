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

import gate.Annotation;
import gate.Corpus;
import gate.Document;
import org.apache.commons.io.FilenameUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.io.UnsupportedEncodingException;
import java.util.Iterator;

/**
 * User: hfriedrich
 * Date: 01.07.2014
 *
 * Create data (binary tensor) for the RESCAL algorithm from a corpus of text files.
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

  private WonMatchingData matchingData;

  public GateRESCALProcessing() {
    matchingData = new WonMatchingData();
  }

  /**
   * Add a Gate-processed corpus to the class to generate output of it later by {@link #createRescalData(String)}.
   * The documents in the corpus must have been annotated correctly by Gate (see annotation definition constants in
   * this class).
   *
   * @param corpus
   */
  public void addDataFromProcessedCorpus(Corpus corpus) throws UnsupportedEncodingException {
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

  private String createNeedId(Document doc) throws UnsupportedEncodingException {
    return java.net.URLDecoder.decode(FilenameUtils.getBaseName(doc.getSourceUrl().getFile()), "UTF-8");
  }

  /**
   * Save the data that was added using {@link #addDataFromProcessedCorpus(gate.Corpus)}.
   * The data created is binary tensor data saved in a matlab file. Additionally a header file is generated for
   * correlation of indices in the tensor with the original needs/documents.
   *
   * @param outputFolder data folder
   * @throws IOException
   */
  public void createRescalData(String outputFolder) throws IOException {
    matchingData.writeOutputFiles(outputFolder);
  }

}
