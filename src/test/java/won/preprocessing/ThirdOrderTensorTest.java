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

/**
 * User: hfriedrich
 * Date: 09.07.2014
 */
public class ThirdOrderTensorTest
{
  private static final double DELTA = 0.001d;

  private ThirdOrderTensor testTensor1;

  @Before
  public void initTestTensor() {
    testTensor1 = new ThirdOrderTensor(4, 4, 3, 10);
  }

  @Test
  public void tensorCreation() {

    ThirdOrderTensor tensor = new ThirdOrderTensor(4, 3, 2, 10);
    int[] dim = {4, 3, 2};
    Assert.assertArrayEquals(tensor.getDimensions(), dim);
    for (int x3 = 0; x3 < dim[2]; x3++) {
      for (int x2 = 0; x2 < dim[1]; x2++) {
        for (int x1 = 0; x1 < dim[0]; x1++) {
          Assert.assertEquals(0.0d, tensor.getEntry(x1, x2, x3), 0.0d);
        }
      }
    }
  }

  @Test
  public void setGetEntry() {

    testTensor1.setEntry(0.5d, 0, 0, 0);
    testTensor1.setEntry(1.0d, 0, 0, 0);
    testTensor1.setEntry(2.0d, 1, 0, 1);
    testTensor1.setEntry(3.0d, 0, 2, 2);
    testTensor1.setEntry(4.0d, 3, 3, 2);

    Assert.assertEquals(1.0d, testTensor1.getEntry(0, 0, 0), DELTA);
    Assert.assertEquals(2.0d, testTensor1.getEntry(1, 0, 1), DELTA);
    Assert.assertEquals(3.0d, testTensor1.getEntry(0, 2, 2), DELTA);
    Assert.assertEquals(4.0d, testTensor1.getEntry(3, 3, 2), DELTA);
  }

  @Test
  public void resizeUp() {

    int[] dim = testTensor1.getDimensions();
    testTensor1.setEntry(1.0d, 3, 1, 2);
    int[] newDim = {dim[0]+1, dim[1]+2, dim[2]+3};
    testTensor1.resize(newDim[0], newDim[1], newDim[2], 10);
    Assert.assertArrayEquals(newDim, testTensor1.getDimensions());
    Assert.assertEquals(1.0d, testTensor1.getEntry(3, 1, 2), DELTA);

    for (int x3 = 0; x3 < newDim[2]; x3++) {
      for (int x2 = 0; x2 < newDim[1]; x2++) {
        for (int x1 = 0; x1 < newDim[0]; x1++) {
          if (x1 != 3 || x2 != 1 || x3 != 2) {
            Assert.assertEquals(0.0d, testTensor1.getEntry(x1, x2, x3), 0.0d);
          }
        }
      }
    }
  }

  @Test
  public void resizeDown() {

    int[] dim = testTensor1.getDimensions();
    testTensor1.setEntry(1.0d, 3, 1, 2);
    int[] newDim = {dim[0]-1, dim[1]-1, dim[2]-1};
    testTensor1.resize(newDim[0], newDim[1], newDim[2], 10);
    Assert.assertArrayEquals(newDim, testTensor1.getDimensions());

    for (int x3 = 0; x3 < newDim[2]; x3++) {
      for (int x2 = 0; x2 < newDim[1]; x2++) {
        for (int x1 = 0; x1 < newDim[0]; x1++) {
            Assert.assertEquals(0.0d, testTensor1.getEntry(x1, x2, x3), 0.0d);
        }
      }
    }
  }

}
