__author__ = 'hfriedrich'

import os
import sys
import codecs

# simple script to create a text file with all the mail text file names (needs) in a directory.
# this is used for categorization of the needs, also see script create_categories.py

mailfolder = sys.argv[1]

oldfile = codecs.open("C:/dev/temp/testcorpus/complete/out/rescal/allneeds.txt",  mode="r", encoding="utf8")
oldlines = oldfile.read().splitlines()

mails = [unicode(s, "utf-8", errors="replace") for s in os.listdir(mailfolder)]

out = codecs.open(mailfolder + "/allneeds3.txt", mode="w+", encoding="utf8")



