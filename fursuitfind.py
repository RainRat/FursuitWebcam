import cv2
import csv
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from os import listdir
from os.path import isfile, join, isdir
import argparse

def getAllFilesRecursive(root):
    files = [ join(root,f) for f in listdir(root) if isfile(join(root,f))]
    dirs = [ d for d in listdir(root) if isdir(join(root,d))]
    for d in dirs:
        files_in_d = getAllFilesRecursive(join(root,d))
        if files_in_d:
            for f in files_in_d:
                files.append(join(root,f))
    return files

list_classes = ["file", "numfursuit", "numfursuitconfidence", "fursuitname", "fursuitnameconfidence"]
parser = argparse.ArgumentParser()
parser.add_argument('targdir', help='Directory to search (input)')
parser.add_argument('targcsv', help='Output csv file (output)')
args = parser.parse_args()

outputcsvfile=open(args.targcsv, 'w', newline='')
writer=csv.writer(outputcsvfile) 
writer.writerow(list_classes)

csv_reader = csv.reader(open('fursuitlookup.csv'))
fursuitnames=[]
for row in csv_reader:
    fursuitnames.append(row)
    
modelfursuitname = load_model('fursuitname.h5')
modelfursuitcount = load_model('fursuitcount.h5')
fndict=getAllFilesRecursive(args.targdir)        
for fn in fndict:
    print(fn)
    frame = cv2.imread(fn)
    squareframe=cv2.resize(frame, (224,224))
    test_image = cv2.cvtColor(squareframe, cv2.COLOR_BGR2RGB)
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    test_image=test_image/255.0
    
    fursuitnameresult = (modelfursuitname.predict(test_image, batch_size=1))[0]
    maxconname=0.
    for idx, val in enumerate(fursuitnameresult):
        if val>maxconname:
            maxidxname=idx
            maxconname=val
                
    fursuitcountresult = (modelfursuitcount.predict(test_image, batch_size=1))[0]
    fursuitcountnames=[0, 1, 2, 3]
    maxcon=0.
    for idx, val in enumerate(fursuitcountresult):
        if val>maxcon:
            maxidx=idx
            maxcon=val
    writer.writerow([fn, fursuitcountnames[maxidx], maxcon, fursuitnames[maxidxname][1], maxconname])
