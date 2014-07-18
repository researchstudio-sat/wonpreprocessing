package won.preprocessing;/*
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

import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;

/**
 * User: hfriedrich
 * Date: 18.07.2014
 */
public class WonMatchingDataTest
{
  private static final double DELTA = 0.001d;

  private WonMatchingData data;

  @Before
  public void initData() {
    data = new WonMatchingData();
  }

  @Test
  public void dataInitialized() {
    Assert.assertEquals(data.getAttributes().size(), 0);
    Assert.assertEquals(data.getNeeds().size(), 0);
  }

  @Test
  public void addNeedType() {

    data.addNeedType("Need1", WonMatchingData.NeedType.OFFER);
    Assert.assertEquals(data.getNeeds().size(), 1);
    Assert.assertEquals(data.getAttributes().size(), 1);
    Assert.assertTrue(data.getNeeds().contains("Need1"));
    Assert.assertTrue(data.getAttributes().contains("OFFER"));

    data.addNeedType("Need1", WonMatchingData.NeedType.OFFER);
    Assert.assertEquals(data.getNeeds().size(), 1);
    Assert.assertEquals(data.getAttributes().size(), 1);
    Assert.assertTrue(data.getNeeds().contains("Need1"));
    Assert.assertTrue(data.getAttributes().contains("OFFER"));

    data.addNeedType("Need2", WonMatchingData.NeedType.WANT);
    Assert.assertEquals(data.getNeeds().size(), 2);
    Assert.assertEquals(data.getAttributes().size(), 2);
    Assert.assertTrue(data.getNeeds().contains("Need1"));
    Assert.assertTrue(data.getNeeds().contains("Need2"));
    Assert.assertTrue(data.getAttributes().contains("OFFER"));
    Assert.assertTrue(data.getAttributes().contains("WANT"));
  }

  @Test
  public void addNeedConnection() {

    data.addNeedConnection("Need1", "Need2");
    Assert.assertEquals(data.getNeeds().size(), 2);
    Assert.assertEquals(data.getAttributes().size(), 0);
    Assert.assertTrue(data.getNeeds().contains("Need1"));
    Assert.assertTrue(data.getNeeds().contains("Need2"));

    data.addNeedConnection("Need1", "Need3");
    Assert.assertEquals(data.getNeeds().size(), 3);
    Assert.assertEquals(data.getAttributes().size(), 0);
    Assert.assertTrue(data.getNeeds().contains("Need1"));
    Assert.assertTrue(data.getNeeds().contains("Need2"));
    Assert.assertTrue(data.getNeeds().contains("Need3"));
  }

  @Test
  public void addNeedAttribute() {

    data.addNeedAttribute("Need1", "Attr1", WonMatchingData.AttributeType.TOPIC);
    Assert.assertEquals(data.getNeeds().size(), 1);
    Assert.assertEquals(data.getAttributes().size(), 1);
    Assert.assertTrue(data.getNeeds().contains("Need1"));
    Assert.assertTrue(data.getAttributes().contains("Attr1"));

    data.addNeedAttribute("Need1", "Attr2", WonMatchingData.AttributeType.DESCRIPTION);
    Assert.assertEquals(data.getNeeds().size(), 1);
    Assert.assertEquals(data.getAttributes().size(), 2);
    Assert.assertTrue(data.getNeeds().contains("Need1"));
    Assert.assertTrue(data.getAttributes().contains("Attr1"));
    Assert.assertTrue(data.getAttributes().contains("Attr2"));

    data.addNeedAttribute("Need2", "Attr1", WonMatchingData.AttributeType.TOPIC);
    Assert.assertEquals(data.getNeeds().size(), 2);
    Assert.assertEquals(data.getAttributes().size(), 2);
    Assert.assertTrue(data.getNeeds().contains("Need1"));
    Assert.assertTrue(data.getNeeds().contains("Need2"));
    Assert.assertTrue(data.getAttributes().contains("Attr1"));
    Assert.assertTrue(data.getAttributes().contains("Attr2"));
  }

  @Test
  public void checkTensor() throws IOException {

    data.addNeedType("Need1", WonMatchingData.NeedType.OFFER);
    data.addNeedAttribute("Need1", "Couch", WonMatchingData.AttributeType.TOPIC);
    data.addNeedAttribute("Need1", "IKEA", WonMatchingData.AttributeType.TOPIC);
    data.addNeedAttribute("Need1", "...", WonMatchingData.AttributeType.DESCRIPTION);
    data.addNeedType("Need2", WonMatchingData.NeedType.WANT);
    data.addNeedAttribute("Need2", "Leather", WonMatchingData.AttributeType.TOPIC);
    data.addNeedAttribute("Need2", "Couch", WonMatchingData.AttributeType.TOPIC);
    data.addNeedAttribute("Need2", "IKEA", WonMatchingData.AttributeType.DESCRIPTION);
    data.addNeedConnection("Need1", "Need2");
    data.addNeedType("Need3", WonMatchingData.NeedType.WANT);

    ThirdOrderTensor tensor = data.createFinalTensor();
    int[] dim = {9, 9, WonMatchingData.SliceTypes.values().length};
    Assert.assertArrayEquals(dim, tensor.getDimensions());

    Assert.assertEquals(1.0d, tensor.getEntry(0, 1, WonMatchingData.SliceTypes.IS_NEED_TYPE.ordinal()), DELTA);
    Assert.assertEquals(1.0d, tensor.getEntry(0, 2, WonMatchingData.SliceTypes.HAS_TOPIC_ATTRIBUTE.ordinal()), DELTA);
    Assert.assertEquals(1.0d, tensor.getEntry(0, 3, WonMatchingData.SliceTypes.HAS_TOPIC_ATTRIBUTE.ordinal()), DELTA);
    Assert.assertEquals(1.0d, tensor.getEntry(0, 4, WonMatchingData.SliceTypes.HAS_DESCRIPTION_ATTRIBUTE.ordinal()),
                        DELTA);
    Assert.assertEquals(1.0d, tensor.getEntry(5, 6, WonMatchingData.SliceTypes.IS_NEED_TYPE.ordinal()), DELTA);
    Assert.assertEquals(1.0d, tensor.getEntry(5, 7, WonMatchingData.SliceTypes.HAS_TOPIC_ATTRIBUTE.ordinal()), DELTA);
    Assert.assertEquals(1.0d, tensor.getEntry(5, 2, WonMatchingData.SliceTypes.HAS_TOPIC_ATTRIBUTE.ordinal()), DELTA);
    Assert.assertEquals(1.0d, tensor.getEntry(5, 3, WonMatchingData.SliceTypes.HAS_DESCRIPTION_ATTRIBUTE.ordinal()),
                        DELTA);
    Assert.assertEquals(1.0d, tensor.getEntry(0, 5, WonMatchingData.SliceTypes.HAS_CONNECTION.ordinal()), DELTA);
    Assert.assertEquals(1.0d, tensor.getEntry(5, 0, WonMatchingData.SliceTypes.HAS_CONNECTION.ordinal()), DELTA);
    Assert.assertEquals(1.0d, tensor.getEntry(8, 6, WonMatchingData.SliceTypes.IS_NEED_TYPE.ordinal()), DELTA);
  }
}
