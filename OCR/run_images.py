import glob
import os

filenames = glob.glob("../crop*.jpg")

for file in filenames:
    os.system("python3 ocr_sample.py " + str(file))

