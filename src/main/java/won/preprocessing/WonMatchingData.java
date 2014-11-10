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

import java.io.*;
import java.util.ArrayList;
import java.util.List;

/**
 * This class builds up the relations between Needs and their attributes. Use the methods {@link #addNeedType(String,
 * won.preprocessing.WonMatchingData.NeedType)}, {@link #addNeedConnection(String,
 * String)} and {@link #addNeedAttribute(String, String, won.preprocessing.WonMatchingData.AttributeType)} to build
 * an internal data structure (RESCAL three-way-tensor) that refelcts the relations between Needs and attributes.
 *
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
    HAS_CONNECTION("connection"),
    IS_NEED_TYPE("needtype"),
    HAS_TOPIC_ATTRIBUTE("subject"),
    HAS_DESCRIPTION_ATTRIBUTE("content");
    private String sliceFileName;

    private SliceTypes(String fileName) {
      sliceFileName = fileName;
    }

    public String getSliceFileName() {
      return sliceFileName;
    }
  }

  private static final Logger logger = LoggerFactory.getLogger(WonMatchingData.class);

  private static final int MAX_DIMENSION = 1000000;
  private static final String NEED_PREFIX = "Need: ";
  private static final String ATTRIBUTE_PREFIX = "Attr: ";
  private static final String HEADERS_FILE = "headers.txt";
  private static final String DATA_FILE_PREFIX = "data";

  private ThirdOrderSparseTensor tensor;
  private ArrayList<String> needs;
  private ArrayList<String> attributes;
  private int nextIndex = 0;

  public WonMatchingData() {

    int dim = MAX_DIMENSION;
    tensor = new ThirdOrderSparseTensor(dim, dim, SliceTypes.values().length, 1);
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
  }

  public void addNeedType(String need, NeedType type) {

    checkName(need);
    int x1 = addNeed(need);
    int x2 = addAttribute(type.toString());
    tensor.setEntry(1.0d, x1, x2, SliceTypes.IS_NEED_TYPE.ordinal());
  }

  public void addNeedAttribute(String need, String attribute, AttributeType attrType) {

    checkName(need);
    checkName(attribute);
    int x1 = addNeed(need);
    int x2 = addAttribute(attribute);
    int x3 = AttributeType.TOPIC.equals(attrType) ? SliceTypes.HAS_TOPIC_ATTRIBUTE.ordinal() : SliceTypes
      .HAS_DESCRIPTION_ATTRIBUTE.ordinal();
    tensor.setEntry(1.0d, x1, x2, x3);
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

    if (name == null || name.equals("")) {
      throw new IllegalArgumentException("Need/Attribute is not allowed to be null or empty");
    }
  }

  protected ThirdOrderSparseTensor createFinalTensor() {

    int dim = getNeeds().size() + getAttributes().size();

    int maxNZ = 0;
    for (SliceTypes types : SliceTypes.values()) {
      maxNZ = Math.max(tensor.getNonZeroEntries(types.ordinal()), maxNZ);
    }
    tensor.resize(dim, dim, SliceTypes.values().length, maxNZ);
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

    tensor.writeSliceToFile(folder + "/" + SliceTypes.HAS_CONNECTION.getSliceFileName() + ".mtx",
                            SliceTypes.HAS_CONNECTION.ordinal());
    tensor.writeSliceToFile(folder + "/" + SliceTypes.IS_NEED_TYPE.getSliceFileName() + ".mtx",
                            SliceTypes.IS_NEED_TYPE.ordinal());
    tensor.writeSliceToFile(folder + "/" + SliceTypes.HAS_TOPIC_ATTRIBUTE.getSliceFileName() + ".mtx",
                            SliceTypes.HAS_TOPIC_ATTRIBUTE.ordinal());
    tensor.writeSliceToFile(folder + "/" + SliceTypes.HAS_DESCRIPTION_ATTRIBUTE.getSliceFileName() + ".mtx",
                            SliceTypes.HAS_DESCRIPTION_ATTRIBUTE.ordinal());

    // write the headers file
    FileOutputStream fos = new FileOutputStream(new File(folder + "/" + HEADERS_FILE));
    OutputStreamWriter os = new OutputStreamWriter(fos, "UTF-8");

    for (int i = 0; i < nextIndex; i++) {
      String entity = (needs.get(i) != null) ? NEED_PREFIX + needs.get(i) : ATTRIBUTE_PREFIX + attributes.get(i);
      os.append(entity + "\n");
    }
    os.close();

    logger.info("- needs: {}", getNeeds().size());
    logger.info("- attributes: {}", getAttributes().size());
    logger.info("- connections: {}", tensor.getNonZeroEntries(SliceTypes.HAS_CONNECTION.ordinal()) / 2);
    logger.info("- tensor size: {} x {} x " + tensor.getDimensions()[2], tensor.getDimensions()[0],
                tensor.getDimensions()[1]);
  }
}
