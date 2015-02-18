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

import org.la4j.Matrices;
import org.la4j.matrix.sparse.CCSMatrix;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;

/**
 * Sparse third order tensor based on la4j implementation of sparse matrices.
 *
 * User: hfriedrich
 * Date: 09.07.2014
 */
public class ThirdOrderSparseTensor
{
  private CCSMatrix[] slices;
  private int[] dims;

  public ThirdOrderSparseTensor(int dimX1, int dimX2, int dimX3, int nzMaxPerSlice) {

    dims = null;
    slices = null;
    resize(dimX1, dimX2, dimX3, nzMaxPerSlice);
  }

  public void resize(int dimX1, int dimX2, int dimX3, int nzMaxPerSlice) {

    CCSMatrix[] newSlices = new CCSMatrix[dimX3];
    for (int x3 = 0; x3 < dimX3; x3++) {
      if (dims != null && x3 < dims[2]) {
        newSlices[x3] = slices[x3].copyOfShape(dimX1, dimX2).to(Matrices.CCS);
      } else {
        newSlices[x3] = CCSMatrix.zero(dimX1, dimX2);
      }
    }
    dims = new int[]{dimX1, dimX2, dimX3};
    slices = newSlices;
  }

  public void setEntry(double value, int x1, int x2, int x3) {
    slices[x3].set(x1, x2, value);
  }

  public double getEntry(int x1, int x2, int x3) {
    return slices[x3].get(x1, x2);
  }

  public int getNonZeroEntries(int dimX3) {
    return slices[dimX3].cardinality();
  }

  public int[] getDimensions() {
    return dims;
  }

  public void writeToFile(String folder, String filePrefix) throws IOException {
    int i = 0;
    for (CCSMatrix slice : slices) {
      OutputStream os = new FileOutputStream(new File(folder + filePrefix + "-" + i + ".mtx"));
      os.write(slice.toMatrixMarket().getBytes());
      i++;
    }
  }

  public void writeSliceToFile(String fileName, int slice) throws IOException {
    OutputStream os = new FileOutputStream(new File(fileName));
    os.write(slices[slice].toMatrixMarket().getBytes());
  }
}
