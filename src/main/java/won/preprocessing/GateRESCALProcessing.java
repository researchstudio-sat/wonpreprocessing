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

import com.jmatio.io.MatFileWriter;
import com.jmatio.types.MLArray;
import com.jmatio.types.MLUInt8;
import gate.Annotation;
import gate.Corpus;
import gate.Document;
import org.apache.commons.io.FilenameUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.UnsupportedEncodingException;
import java.util.*;

/**
 * User: hfriedrich
 * Date: 01.07.2014
 *
 * Create data (binary tensor) for the RESCAL algorithm from a corpus of text files.
 */
public class GateRESCALProcessing
{
  private static final Logger logger = LoggerFactory.getLogger(GateRESCALProcessing.class);

  public static final String ATTRIBUTE_ANNOTATION = "SubjectToken";
  public static final String FEATURE_VALUE = "string";

  private ArrayList<String> needList;
  private Map<String, Set<String>> attributeNeedMap;

  public GateRESCALProcessing() {
    attributeNeedMap = new TreeMap<String, Set<String>>();
    needList = new ArrayList<String>();
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
      for (Annotation annotation : currDoc.getAnnotations()) {
        if (annotation.getType().equals(ATTRIBUTE_ANNOTATION)) {
          String attrValue = (String) annotation.getFeatures().get(FEATURE_VALUE);
          if (attrValue == null) {
            logger.error("Feature value '{}' not found in annotation '{}'", FEATURE_VALUE,
                         annotation.getId());
          } else {
            String needId = java.net.URLDecoder.decode(FilenameUtils.getBaseName(currDoc.getSourceUrl().getFile()),
                                                                         "UTF-8");
            addAttributeNeedPairToMap(attrValue, needId);
          }
        }
      }
    }
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

    logger.info("create RESCAL data in folder: {}", outputFolder);
    File outFolder = new File(outputFolder);
    outFolder.mkdirs();

    // map the Needs and attributes both as entities in the RESCAL tensor
    int numEntities = attributeNeedMap.keySet().size() + needList.size();
    int[] dims = {numEntities, numEntities, 1};
    logger.info("- needs: {}", needList.size());
    logger.info("- attributes: {}", attributeNeedMap.keySet().size());
    logger.info("- tensor size: {} x {} x " + dims[2], numEntities, numEntities);

    // write the header file (first needs, then attributes)
    FileWriter fw = new FileWriter(new File(outputFolder + "/" + "headers.txt"));
    for (String need : needList) {
      fw.append("Need: " + need + "\n");
    }

    for (String attr : attributeNeedMap.keySet()) {
      fw.append("Attr: " + attr + "\n");
    }
    fw.close();

    // create the tensor
    MLUInt8 mluInt8 = new MLUInt8("Rs", dims);
    int attrIndex = needList.size();
    for (String attr : attributeNeedMap.keySet()) {
      for (String need : attributeNeedMap.get(attr)) {
        int needIndex = needList.indexOf(need);
        int tensorIndex = attrIndex * dims[0] + needIndex;
        mluInt8.set((byte) 1, tensorIndex);
      }
      attrIndex++;
    }

    // write the output matlab file
    MatFileWriter matWriter = new MatFileWriter();
    final ArrayList<MLArray> list = new ArrayList<MLArray>();
    list.add(mluInt8);
    matWriter.write(new File(outputFolder + "/" + "tensordata.mat"), list);
  }

  private void addAttributeNeedPairToMap(String attribute, String need) {
    Set<String> needs = attributeNeedMap.get(attribute);
    if (needs == null) {
      needs = new TreeSet<String>();
      attributeNeedMap.put(attribute, needs);
    }
    needs.add(need);
    if (!needList.contains(need)) {
      needList.add(need);
    }
  }
}
