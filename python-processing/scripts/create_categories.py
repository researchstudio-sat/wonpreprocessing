__author__ = 'hfriedrich'

import os
import sys
import codecs

# simple script that takes a text file of needs that are categorized and creates output text files for each category
# in which the needs that belong to a certain category are listed. This can be used to easier create connections
# between needs manually then.
# sample needs categorization file entries:
#
# ...
# sport: [FCNYC] WANTED  Childrens Skis and Boots  (Queens) - 'ecomajor' (shidduch@gmail.com) - 2014-02-24 1131
# kitchen_dishes, decoration: [FCNYC] WANTED  China Ware (Lower East Side) - Hadiya Peele (hdypl@yahoo.com) - 2014-01-22 1818
# decoration_xmas, kitchen_baking: [FCNYC] Wanted  Christmas Cake Molds 10461 - Jenny Carchi (jcteacher@ymail.com) - 2014-01-01 0339
# decoration_xmas, decoration: [FCNYC] WANTED  christmas lights, led lights (Brooklyn, NY) - VEE (viveent@gmail.com) - 2014-04-04 2215
# movie_player: [FCNYC] WANTED  Clean Working VCR with remote  (Manhattan) - laronowitz@aol.com - 2014-03-31 0414
# household, storage: [FCNYC] Wanted  Clear plastic storage bins with  lids (Brooklyn and Manhattan) - Midwood Girl (girlmidwood@yahoo.com) - 2014-02-25 1230
# craft_material, office: [FCNYC] WANTED  Clear Plexiglass Sheets [All Boroughs] - (candleshoeev@yahoo.com) - 2014-04-25 2112
# book: [FCNYC] WANTED  Close to the bone David Legge ISBN 9780957739222 ( Lindenwood_Howard Beach) - 'wilrose3' (wilrose3@moxamia.com) - 2014-05-01 1344
# ...
#
# => text files for all the categories (sport, kitchen_dishes, decoration, ...) will be created with the related neeeds

folder = sys.argv[1]
outfolder = folder + "/categories/"
file = codecs.open(folder + "/allneeds.txt",  mode="r", encoding="utf8")
lines = file.read().splitlines()

if not os.path.exists(outfolder):
    os.makedirs(outfolder)

numNeeds = 0
for line in lines:
    temp = line.split(":")
    categories = temp[0]
    if len(temp) > 1:
        need = temp[1].lstrip()
        numNeeds = numNeeds + 1
        categories = categories.split(",")
        for cat in categories:
            cat = cat.lstrip();
            cat = cat.rstrip();
            cat_file = codecs.open(outfolder + cat + ".txt",  mode="a+", encoding="utf8")
            cat_file.write(need + "\n")
            cat_file.close()

print "%d Needs processed." % numNeeds