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

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * User: hfriedrich
 * Date: 17.07.2014
 */
public class WonMatchingData
{
  public enum NeedType {
    OFFER,
    WANT
  }

  public enum AttributeType {
    TOPIC,
    DESCRIPTION
  }

  public enum SliceTypes {
    HAS_CONNECTION,
    IS_NEED_TYPE,
    HAS_TOPIC_ATTRIBUTE,
    HAS_DESCRIPTION_ATTRIBUTE
  }

  private static final Logger logger = LoggerFactory.getLogger(WonMatchingData.class);

  private static final int MAX_DIMENSION = 1000000;
  private static final String NEED_PREFIX = "Need: ";
  private static final String ATTRIBUTE_PREFIX = "Attr: ";
  private static final String DATA_FILE = "data.mat";
  private static final String HEADERS_FILE = "headers.txt";

  private ThirdOrderTensor tensor;
  private ArrayList<String> needs;
  private ArrayList<String> attributes;
  private int nextIndex = 0;
  private int maxNonZeros = 0;

  public WonMatchingData() {

    int dim = MAX_DIMENSION;
    tensor = new ThirdOrderTensor(dim, dim, SliceTypes.values().length, 1);
    needs = new ArrayList<String>();
    attributes = new ArrayList<String>();
  }

  public void addNeedConnection(String need1, String need2) {

    checkName(need1);
    checkName(need2);
    int x1 = addNeed(need1);
    int x2 = addNeed(need2);
    tensor.setEntry(1.0d, x1, x2, SliceTypes.HAS_CONNECTION.ordinal());
    tensor.setEntry(1.0d, x2, x1, SliceTypes.HAS_CONNECTION.ordinal());
    maxNonZeros++;
    maxNonZeros++;
  }

  public void addNeedType(String need, NeedType type) {

    checkName(need);
    int x1 = addNeed(need);
    int x2 = addAttribute(type.toString());
    tensor.setEntry(1.0d, x1, x2, SliceTypes.IS_NEED_TYPE.ordinal());
    tensor.setEntry(1.0d, x2, x1, SliceTypes.IS_NEED_TYPE.ordinal());
    maxNonZeros++;
    maxNonZeros++;
  }

  public void addNeedAttribute(String need, String attribute, AttributeType attrType) {

    checkName(need);
    checkName(attribute);
    int x1 = addNeed(need);
    int x2 = addAttribute(attribute);
    int x3 = AttributeType.TOPIC.equals(attrType) ? SliceTypes.HAS_TOPIC_ATTRIBUTE.ordinal() : SliceTypes
      .HAS_DESCRIPTION_ATTRIBUTE.ordinal();
    tensor.setEntry(1.0d, x1, x2, x3);
    tensor.setEntry(1.0d, x2, x1, x3);
    maxNonZeros++;
    maxNonZeros++;
  }

  private int addNeed(String need) {
    if (!needs.contains(need)) {
      needs.add(nextIndex, need);
      attributes.add(nextIndex, null);
      nextIndex++;
    }
    return needs.indexOf(need);
  }

  private int addAttribute(String attr) {
    if (!attributes.contains(attr)) {
      attributes.add(nextIndex, attr);
      needs.add(nextIndex, null);
      nextIndex++;
    }
    return attributes.indexOf(attr);
  }

  private void checkName(String name) {
    if (NeedType.OFFER.name().equals(name) || NeedType.WANT.name().equals(name)) {
      throw new IllegalArgumentException("Need/Attribute is not allowed to be named like keyword 'OFFER' or 'WANT'");
    }
  }

  protected ThirdOrderTensor createFinalTensor() {

    int dim = getNeeds().size() + getAttributes().size();
    tensor.resize(dim, dim, SliceTypes.values().length, maxNonZeros);
    return tensor;
  }

  public List<String> getNeeds() {
    ArrayList<String> continuousList = new ArrayList<String>();
    for (String need : needs) {
      if (need != null) {
        continuousList.add(need);
      }
    }
    return continuousList;
  }

  public List<String> getAttributes() {
    ArrayList<String> continuousList = new ArrayList<String>();
    for (String attr : attributes) {
      if (attr != null) {
        continuousList.add(attr);
      }
    }
    return continuousList;
  }

  public void writeOutputFiles(String folder) throws IOException {

    File outFolder = new File(folder);
    outFolder.mkdirs();
    if (!outFolder.isDirectory()) {
      return;
    }

    logger.info("create RESCAL data in folder: {}", folder);

    // write the data file
    createFinalTensor();
    int dim = tensor.getDimensions()[0];
    if (dim > MAX_DIMENSION) {
      logger.error("Maximum Dimension {} exceeded: {}", MAX_DIMENSION, dim);
      return;
    }

    tensor.writeToFile(new File(folder + "/" + DATA_FILE));

    // write the headers file
    FileWriter fw = new FileWriter(new File(folder + "/" + HEADERS_FILE));
    for (int i = 0; i < nextIndex; i++) {
      String entity = (needs.get(i) != null) ? NEED_PREFIX + needs.get(i) : ATTRIBUTE_PREFIX + attributes.get(i);
      fw.append(entity + "\n");
    }
    fw.close();

    logger.info("- needs: {}", getNeeds().size());
    logger.info("- attributes: {}", getAttributes().size());
    logger.info("- tensor size: {} x {} x " + tensor.getDimensions()[2], tensor.getDimensions()[0],
                tensor.getDimensions()[1]);
  }

}
