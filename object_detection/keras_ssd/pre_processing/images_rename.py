import os
import sys

path = os.path.join(sys.argv[1], '')

for imagename in os.listdir(path):
    os.rename(path + imagename, path + imagename.replace(".resized", ""))
