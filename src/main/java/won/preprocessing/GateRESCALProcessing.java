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
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.*;

/**
 * User: hfriedrich
 * Date: 01.07.2014
 */
public class GateRESCALProcessing
{
  private static final Logger logger = LoggerFactory.getLogger(GateRESCALProcessing.class);

  public static final String ATTRIBUTE_ANNOTATION = "SubjectToken";
  public static final String FEATURE_VALUE = "string";

  private Map<String, Set<String>> attributeEntityMap;

  public GateRESCALProcessing() {
    attributeEntityMap = new TreeMap<String, Set<String>>();

  }

  public void addDataFromProcessedCorpus(Corpus corpus) {

    Iterator documentIterator = corpus.iterator();
    while (documentIterator.hasNext()) {
      Document currDoc = (Document) documentIterator.next();
      for (Annotation annotation : currDoc.getAnnotations()) {
        if (annotation.getType().equals(ATTRIBUTE_ANNOTATION)) {
          String attrValue = (String) annotation.getFeatures().get(FEATURE_VALUE);
          if (attrValue == null) {
            logger.error("Feature value '{}' not found in annotation '{}'", FEATURE_VALUE,
                         annotation.getId());
          } else {
            addAttributeEntityPairToMap(attrValue, annotation.getId().toString());
          }
        }
      }
    }
  }

  public void createOutputData(String outputFolder) {

    logger.info("create RESCAL data in folder: {}", outputFolder);
    File outFolder = new File(outputFolder);
    outFolder.mkdirs();

    // TODO: ...
  }

  private void addAttributeEntityPairToMap(String attribute, String entity) {
    Set<String> entities = attributeEntityMap.get(attribute);
    if (entities == null) {
      entities = new TreeSet<String>();
      attributeEntityMap.put(attribute, entities);
    }
    entities.add(entity);
  }

}
