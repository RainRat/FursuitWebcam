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
import asyncio
import pyautogui

SCREEN_USAGE=.9
LABEL_FONT=cv2.FONT_HERSHEY_SIMPLEX
LABEL_FRAMES=5
LABEL_SCALE=1.5
LABEL_COLOR=(0, 0, 255)
LABEL_THICKNESS=2

async def main():
    #set up Telegram
    fursuitwebcambot=''
    try:
        with open('apikey.txt', 'r') as file:
            fursuitwebcambot = file.readline().strip()
            targetgroup = int(file.readline().strip())
            bot = telegram.Bot(token=fursuitwebcambot)
    except FileNotFoundError:
        print("Telegram API key file not found. Running in local-only mode...")

    #set up models
    modelmode='i' #"c"=rainratcrowdsourcebot, "i"=imagenet
    mobile = MobileNetV2()
    csv_reader = csv.reader(open('fursuitlookup.csv', encoding='utf8'))
    justfursuitnames = [row[1] for row in csv_reader]

    modelfursuitname = load_model('fursuitname.h5')

    #set up webcam
    screen_width, screen_height=pyautogui.size()
    sleeptime=.1
    pausemode=False
    cap = cv2.VideoCapture(0)
    preds=0
    command_acknowledgement_counter=0

    while True:
        cvw=cv2.waitKey(1) & 0xFF
        acknowledge=True
        if cvw == ord('q'):
            break
        elif (cvw == ord('d') or cvw == ord('`') or cvw == ord('+')) and fursuitwebcambot!='':
            cv2.imwrite('tempimg.png', frame)
            try:
                command_acknowledgement="Sent photo to Telegram"
                await bot.send_photo(chat_id=targetgroup, photo=open('tempimg.png', 'rb'))
            except Exception:
                command_acknowledgement="Failed to send photo to Telegram"
        elif cvw == ord('i'): 
            modelmode='i'
            command_acknowledgement="Switched to ImageNet"
        elif cvw == ord('c'): 
            modelmode='c'        
            command_acknowledgement="Switched to custom"
        elif cvw == ord('s'): 
            sleeptime=sleeptime*1.5   
            command_acknowledgement="Slower"
        elif cvw == ord('f'): 
            sleeptime=sleeptime*0.75
            command_acknowledgement="Faster"
        elif cvw == ord('p'): 
            pausemode=not (pausemode)
            command_acknowledgement="Pause/Unpause"
        else:
            acknowledge=False
        if acknowledge:
            command_acknowledgement_counter=LABEL_FRAMES
        elif pausemode:
            continue
        preds+=1
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

        frame_height, frame_width, _ =frame.shape
        target_height = screen_height * SCREEN_USAGE
        frame=cv2.resize(frame, (int(target_height*(frame_width/frame_height)), int(target_height)))
        for i in range(5):
            label = labelname[i] + " " + '{percent:.0%}'.format(percent=(confidence[i]))
            cv2.putText(frame, label, (20, 75 + i*50), LABEL_FONT, LABEL_SCALE, LABEL_COLOR, LABEL_THICKNESS)
        if command_acknowledgement_counter > 0:
            command_acknowledgement_counter -= 1
            text_size, _ = cv2.getTextSize(command_acknowledgement, LABEL_FONT, LABEL_SCALE, LABEL_THICKNESS)
            text_x = (frame.shape[1] - text_size[0]) // 2
            text_y = (frame.shape[0] - text_size[1]) // 2
            cv2.putText(frame, command_acknowledgement, (text_x, text_y), LABEL_FONT, LABEL_SCALE, LABEL_COLOR, LABEL_THICKNESS)

        cv2.imshow('frame', frame)
        sleep(sleeptime)
asyncio.run(main())