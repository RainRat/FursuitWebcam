import cv2
import csv
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np


csv_reader = csv.reader(open('fursuitlookup.csv'))
fursuitnames=[]
for row in csv_reader:
    fursuitnames.append(row)
    
modelfursuitname = load_model('fursuitname.h5')
modelfursuitcount = load_model('fursuitcount.h5')
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    squareframe=cv2.resize(frame, (224,224))
    test_image = cv2.cvtColor(squareframe, cv2.COLOR_BGR2RGB)
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    test_image=test_image/255.0
    
    fursuitnameresult = (modelfursuitname.predict(test_image, batch_size=1))[0]
    maxcon=0.
    for idx, val in enumerate(fursuitnameresult):
        if val>maxcon:
            maxidx=idx
            maxcon=val
#    print(maxidx)
    thereply=fursuitnames[maxidx][1]+ ": "+'{percent:.0%}'.format(percent=(maxcon))
                
    fursuitcountresult = (modelfursuitcount.predict(test_image, batch_size=1))[0]
    fursuitcountnames=["0", "1", "2", "3+"]
    maxcon=0.
    for idx, val in enumerate(fursuitcountresult):
        if val>maxcon:
            maxidx=idx
            maxcon=val
    thereply2="# of fursuits: "+fursuitcountnames[maxidx]+ " ("+'{percent:.0%}'.format(percent=(maxcon))+")"



    frame=cv2.resize(frame, (1024, 780))
#    cv2.putText(frame, thereply, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    cv2.putText(frame, thereply, (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
    cv2.putText(frame, thereply2, (20, 740), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
        
        
