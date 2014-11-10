__author__ = 'hfriedrich'

import sys
import codecs
import numpy as np
from scipy.sparse import csr_matrix
from scipy.io import mmwrite

# simple script that creates a categoryslice.mtx from the headers.txt file and the categorized needs in the
# allneeds.txt file. In case the file format of headers.txt and allneeds.txt differs, some file names have to be
# adjusted manually in the allneeds.txt.
# Also after the creation was successful the other data-0,1,2.mtx files have to be adjusted to have the same
# dimension as the generated categoryslice.mtx if they are read and evaluated together.

import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S')
_log = logging.getLogger('createcategoryslice.py')


folder = sys.argv[1]
allneedsfile = folder + "/allneeds.txt"
headerfile = folder + "/headers.txt"


_log.info("Read header input file: " + headerfile)
input = codecs.open(headerfile,'r',encoding='utf8')
headers = input.read().splitlines()
input.close()

_log.info("Read allneeds input file with categories: " + allneedsfile)
input = codecs.open(allneedsfile,'r',encoding='utf8')
allneeds = input.read().splitlines()
input.close()

MAX_CATEGORIES = 1000
size = len(headers) + MAX_CATEGORIES
category_slice = np.zeros(shape=(size,size))
categories = []

for categorized_need in allneeds:
    categorized_need = categorized_need.split(":")
    need = categorized_need[1].lstrip()
    i = headers.index("Need: " + need)
    cat = categorized_need[0].split(",")
    for category in cat:
        category = category.lstrip()
        category = category.rstrip()
        if category not in categories:
            categories.append(category)
        j = categories.index(category) + len(headers)
        category_slice[i,j] = 1.0

newSize = len(headers) + len(categories)
category_slice = category_slice[:newSize,:newSize]
mmwrite(folder + "/categoryslice.mtx", csr_matrix(category_slice))

_log.info("Write categories to header input file: " + headerfile)
out = codecs.open(headerfile,'a+',encoding='utf8')
for c in categories:
    out.write("Attr: " + c + "\n")
out.close()


