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

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;

/**
 * User: hfriedrich
 * Date: 09.07.2014
 */
public class ThirdOrderTensor
{
  private MLSparseEx[] slices;
  private int[] dims;

  public ThirdOrderTensor(int dimX1, int dimX2, int dimX3, int nzMaxPerSlice) {

    dims = null;
    slices = null;
    resize(dimX1, dimX2, dimX3, nzMaxPerSlice);
  }

  public void resize(int dimX1, int dimX2, int dimX3, int nzMaxPerSlice) {

    MLSparseEx[] newSlices = new MLSparseEx[dimX3];
    for (int x3 = 0; x3 < dimX3; x3++) {
      if (dims != null && x3 < dims[2]) {
        int[] newDims = {dimX1, dimX2};
        slices[x3] = slices[x3].resize(newDims, nzMaxPerSlice);
        newSlices[x3] = slices[x3];
      } else {
        newSlices[x3] = new MLSparseEx("Rs" + x3, new int[]{dimX1, dimX2}, 0, nzMaxPerSlice);
      }
    }
    dims = new int[]{dimX1, dimX2, dimX3};
    slices = newSlices;
  }

  public void setEntry(double value, int x1, int x2, int x3) {
    slices[x3].set(value, x1, x2);
  }

  public double getEntry(int x1, int x2, int x3) {
    return slices[x3].get(x1, x2);
  }

  public int[] getDimensions() {
    return dims;
  }

  public void writeToFile(File file) throws IOException {

    MatFileWriter matWriter = new MatFileWriter();
    final ArrayList<MLArray> list = new ArrayList<MLArray>();
    for (MLSparseEx slice : slices) {
      list.add(slice);
    }
    matWriter.write(file, list);
  }
}
