import cv2
import csv
import gc
import telegram
from tensorflow import expand_dims
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.backend import clear_session
from time import sleep

fursuitwebcambot=''
try:
    with open('apikey.txt', 'r') as file:
        fursuitwebcambot = file.readline().strip()
        targetgroup = int(file.readline().strip())
        bot = telegram.Bot(token=fursuitwebcambot)
except FileNotFoundError:
    print("Telegram API key file not found. Running in local-only mode...")

modelmode='i' #"c"=rainratcrowdsourcebot, "i"=imagenet

mobile = MobileNetV2()

csv_reader = csv.reader(open('fursuitlookup.csv', encoding='utf8'))
fursuitnames=[]
justfursuitnames=[]
for row in csv_reader:
    fursuitnames.append(row)
for i in range (len(fursuitnames)):
    justfursuitnames.append(fursuitnames[i][1])
modelfursuitname = load_model('fursuitname.h5')

sleeptime=.1
pausemode=False
cap = cv2.VideoCapture(0)
preds=0
while True:
    cvw=cv2.waitKey(1) & 0xFF
    if cvw == ord('q'):
        break
    elif cvw == ord('d') and fursuitwebcambot!='':
        cv2.imwrite('tempimg.png', frame)
        bot.send_photo(chat_id=targetgroup, photo=open('tempimg.png','rb'))
    elif cvw == ord('i'): 
        modelmode='i'
    elif cvw == ord('c'): 
        modelmode='c'        
    elif cvw == ord('s'): 
        sleeptime=sleeptime*1.5       
    elif cvw == ord('f'): 
        sleeptime=sleeptime*0.75
    elif cvw == ord('p'): 
        pausemode=not (pausemode)
    if pausemode==True:
        continue
    preds=preds+1
    if preds %500==0: #ran into memory leak in tensorflow
        clear_session()
        gc.collect()
    ret, frame = cap.read()
    squareframe=cv2.resize(frame, (224,224))
    test_image = cv2.cvtColor(squareframe, cv2.COLOR_BGR2RGB)
    test_image = img_to_array(test_image)
    test_image= expand_dims(test_image, axis=0)
    test_image=preprocess_input(test_image) #mobilenet only!!!! 

    if modelmode=='i':
        rawresult = mobile.predict(test_image,verbose = 0)
        results = decode_predictions(rawresult)
        junk, labelname, confidence  = zip(*results[0])
    if modelmode=='c':
        rawresult = modelfursuitname.predict(test_image,verbose = 0)
        zipped_lists=zip(rawresult[0], justfursuitnames)
        sorted_zipped_lists = sorted(zipped_lists, reverse=True)
        confidence, labelname = zip(*sorted_zipped_lists)        

    thereplya=labelname[0]+ " "+'{percent:.0%}'.format(percent=(confidence[0]))
    thereplyb=labelname[1]+ " "+'{percent:.0%}'.format(percent=(confidence[1]))
    thereplyc=labelname[2]+ " "+'{percent:.0%}'.format(percent=(confidence[2]))
    thereplyd=labelname[3]+ " "+'{percent:.0%}'.format(percent=(confidence[3]))
    thereplye=labelname[4]+ " "+'{percent:.0%}'.format(percent=(confidence[4]))

    frame=cv2.resize(frame, (1024, 780))

    cv2.putText(frame, thereplya, (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
    cv2.putText(frame, thereplyb, (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
    cv2.putText(frame, thereplyc, (20, 175), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
    cv2.putText(frame, thereplyd, (20, 225), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
    cv2.putText(frame, thereplye, (20, 275), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)

    cv2.imshow('frame', frame)
    sleep(sleeptime)
