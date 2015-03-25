__author__ = 'hfriedrich'

import os
import sys
import string
import shutil
import re

import logging
logging.basicConfig(level=logging.INFO)
_log = logging.getLogger()

# Simple script to normalize filenames before processing them in different environments (python 2, python 3,
# win/unix, java/gate) which can lead to problems. This should be the first script to execute on the data

def replace_chars(str, niceChars, replaceChar):
    newstring = ""
    for char in list(str):
        newstring += (char if char in niceChars else replaceChar)
    return newstring

NICE_CHARS = list(string.ascii_letters + string.digits + " " +"_-.,:@()[]")

infolder = sys.argv[1]
outfolder = sys.argv[2]
mails = os.listdir(infolder)

if not os.path.exists(outfolder):
    os.makedirs(outfolder)

email_regex = re.compile(r"[a-zA-Z0-9_.-]+@[a-zA-Z0-9_.-]+\.[a-zA-Z0-9_.-]+")

_log.info("Read file names from folder: " + infolder)
_log.info("Normalize file names and copy files to folder: " + outfolder)
for mail in mails:
    if os.path.isfile(infolder + "/" + mail):
        newFileName = replace_chars(mail, NICE_CHARS, "_")
        newFileName = email_regex.sub("xxx@xxx.xx", newFileName)
        shutil.copyfile(infolder + "/" + mail, outfolder + "/" + newFileName)
    elif not os.path.isdir(infolder + "/" + mail):
        _log.warn("Problem with file (must be renamed and copied manually): " + mail)
    else:
        _log.warn("Directory not processed: " + mail)
