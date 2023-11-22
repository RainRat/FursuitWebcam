import cv2
import csv
import gc
import pygame
import torch
import telegram
from tensorflow import expand_dims
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.backend import clear_session
import asyncio
import pyautogui
import base64
from openai import OpenAI
import configparser
import io

SCREEN_USAGE=.9
LABEL_FONT=cv2.FONT_HERSHEY_SIMPLEX
LABEL_FRAMES=25
LABEL_SCALE=1.8
LABEL_COLOR=(0, 0, 255)
LABEL_THICKNESS=2
FPS_INCREMENT = 3
MIN_FPS = 1
MAX_FPS = 30
STICK_USED = 0.8

HAT_MAP = {
  (-1, 0): 'left',
  (1, 0): 'right',
  (0, -1): 'slow',
  (0, 1): 'fast'
}

KEY_MAP = {
  pygame.K_d: 'post',
  pygame.K_KP_PLUS: 'post',
  pygame.K_BACKQUOTE: 'post',
  pygame.K_LEFT: 'left',
  pygame.K_RIGHT: 'right',
  pygame.K_s: 'slow',
  pygame.K_DOWN: 'slow',
  pygame.K_f: 'fast',
  pygame.K_UP: 'fast',
  pygame.K_p: 'pause',
  pygame.K_SPACE: 'pause',
  pygame.K_q: 'quit'
}
BUTTON_MAP= {
  0: 'post',
  1: 'post',
  2: 'post',
  3: 'post',
  8: 'pause'
}

def update_status_bar(screen, model_list_status, modelmode):
    bar_height = 25
    screen_width = screen.get_width()
    section_width = screen_width // len(model_list)
    active_color = (0, 255, 0)
    inactive_color = (255, 255, 255)

    for i, model_name in enumerate(model_list_status):
        color = active_color if i == modelmode else inactive_color
        pygame.draw.rect(screen, color, [i * section_width, 0, section_width, bar_height])
        font = pygame.font.SysFont(None, 24)
        text = font.render(model_name, True, (0, 0, 0))
        screen.blit(text, (i * section_width + 5, 5))

def resize_to(img, target_size):
    h, w = img.shape[:2]
    scale = target_size / max(h, w)
    img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img

def handle_events():
    input_str = None
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            input_str = KEY_MAP.get(event.key)
        elif event.type == pygame.JOYBUTTONDOWN:
            input_str = BUTTON_MAP.get(event.button)
        elif event.type == pygame.JOYHATMOTION:
            input_str = HAT_MAP.get(event.value, '')
        elif event.type == pygame.JOYAXISMOTION:
            if event.axis == 0:
                if event.value < -STICK_USED:
                    input_str = 'left'
                elif event.value > STICK_USED:
                    input_str = 'right'
            elif event.axis == 1:
                if event.value < -STICK_USED:
                    input_str = 'fast'
                elif event.value > STICK_USED:
                    input_str = 'slow'
        elif event.type == pygame.QUIT:
            input_str = 'quit'
    return input_str

async def main():
    pygame.init()
    pygame.joystick.init()
    pygame.mixer.init()
    clock = pygame.time.Clock()
    running = True
    first_loop = True
    cur_fps = 20
    image_cv2 = None
    bot = telegram.Bot(token=tg_key)
    info_bar = ""
    joystick = None
    if pygame.joystick.get_count() > 0:
        joystick = pygame.joystick.Joystick(0)
        joystick.init()
    modelmode = 1
    mobile = MobileNetV2()
    with open(custom_lookup, encoding='utf8') as csv_file:
        csv_reader = csv.reader(csv_file)
        custom_names = [row[1] for row in csv_reader]
    custom_model = load_model(custom_file)
    modelyolo = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    screen_width, screen_height=pyautogui.size()
    pausemode=False
    cap = cv2.VideoCapture(0)
    preds = 0
    info_counter=0

    while running:
        clock.tick(cur_fps)
        input_str=handle_events()
        acknowledge=True
        if input_str == 'quit':
            running= False
            info_bar = "Quitting"
        elif input_str == 'post' and bot:
            caption_str = ''
            shutter_sound = pygame.mixer.Sound('shutter.mp3')
            shutter_sound.play()
            if modelmode == 3 and openai_key:
                try:
                    _, buffer = cv2.imencode(".png", resize_to(image_orig, 512))
                    base64_image = base64.b64encode(buffer).decode('utf-8')
                    client = OpenAI(api_key=openai_key)
                    response = client.chat.completions.create(model=openai_model, max_tokens=800, messages= [
                        { "role": "user", "content": [ { "type": "text", "text": f"{openai_prompt}" },
                          { "type": "image_url", "image_url": { "url": f"data:image/jpeg;base64,{base64_image}" }, }, ], 
                        } ], )
                    caption_str=caption_text + "\r\n" + response.choices[0].message.content
                except Exception as e:
                    print(f"Error: {e}")
                    caption_str = "No caption. Error connecting to GPT."
            try:
                _, buffer = cv2.imencode(".png", image_cv2)
                bio = io.BytesIO(buffer)
                await bot.send_photo(chat_id=tg_group, photo=bio, caption=caption_str)
                info_bar = "Sent photo to Telegram: " + tg_name
            except Exception as e:
                print(f"Error: {e}")
                info_bar = "Failed to send photo to Telegram"

        elif input_str in ('right', 'left'):
            modelmode = (modelmode + (1 if input_str == 'right' else -1)) % len(model_list)
            info_bar="Switched to "+ model_list[modelmode]
        elif input_str == 'slow':
            if cur_fps - FPS_INCREMENT > MIN_FPS:
                cur_fps -= FPS_INCREMENT
                info_bar = "Slower"
            else:
                cur_fps = MIN_FPS
                info_bar = "Slowest"
        elif input_str == 'fast':
            if cur_fps + FPS_INCREMENT < MAX_FPS:
                cur_fps += FPS_INCREMENT
                info_bar = "Faster"
            else:
                cur_fps = MAX_FPS
                info_bar = "Fastest"
        elif input_str=='pause': 
            pausemode=not (pausemode)
            info_bar="Pause/Unpause"
        else:
            acknowledge=False
        if acknowledge:
            info_counter = LABEL_FRAMES
        elif pausemode:
            continue
        preds+=1
        if preds %500==0: #ran into memory leak in tensorflow
            clear_session()
            gc.collect()
        _, image_orig = cap.read()
        image_rgb = cv2.cvtColor(image_orig, cv2.COLOR_BGR2RGB)

        image_mobilenet = cv2.resize(image_rgb, (224,224))
        image_mobilenet = img_to_array(image_mobilenet)
        image_mobilenet = expand_dims(image_mobilenet, axis=0)
        image_mobilenet = preprocess_input(image_mobilenet)

        if modelmode==1:
            rawresult = mobile.predict(image_mobilenet, verbose = 0)
            results = decode_predictions(rawresult)
            _, labelname, confidence = zip(*results[0])
        elif modelmode==0:
            rawresult = custom_model.predict(image_mobilenet, verbose = 0)
            zipped_lists=zip(rawresult[0], custom_names)
            sorted_zipped_lists = sorted(zipped_lists, reverse=True)
            confidence, labelname = zip(*sorted_zipped_lists)
        elif modelmode==2:
            results = modelyolo(image_rgb)
        image_orig_x, image_orig_y, _ = image_orig.shape
        target_height = screen_height * SCREEN_USAGE
        target_width = target_height*(image_orig_y/image_orig_x)
        image_cv2 = cv2.resize(image_orig, (int(target_width), int(target_height)))

        if modelmode in (0, 1):
            for i in range(5):
                label = f"{labelname[i]} {confidence[i]:.0%}"
                cv2.putText(image_cv2, label, (20, 75 + i*50), LABEL_FONT, LABEL_SCALE, LABEL_COLOR, LABEL_THICKNESS)
        elif modelmode ==2:
            bboxes = results.xyxy[0].cpu().numpy()
            scale_x = target_height/image_orig_x
            scale_y = target_width/image_orig_y

            for box in bboxes:
                x1, y1, x2, y2, conf, cls = box
                x1 = x1 * scale_x
                y1 = y1 * scale_y
                x2 = x2 * scale_x
                y2 = y2 * scale_y
                label = results.names[int(cls)]
                label_with_conf = f"{label} ({(conf * 100):.0f}%)"
                cv2.rectangle(image_cv2, (int(x1), int(y1)), (int(x2), int(y2)), LABEL_COLOR, 2)
                cv2.putText(image_cv2, label_with_conf, (int(x1), int(y1) - 10), LABEL_FONT, LABEL_SCALE, LABEL_COLOR, LABEL_THICKNESS)

        if info_counter > 0:
            info_counter -= 1
            text_size, _ = cv2.getTextSize(info_bar, LABEL_FONT, LABEL_SCALE, LABEL_THICKNESS)
            text_x = (image_cv2.shape[1] - text_size[0]) // 2
            text_y = (image_cv2.shape[0] - text_size[1]) // 2
            cv2.putText(image_cv2, info_bar, (text_x, text_y), LABEL_FONT, LABEL_SCALE, LABEL_COLOR, LABEL_THICKNESS)
        image_pygame = cv2.transpose(image_cv2)
        image_pygame = cv2.cvtColor(image_pygame, cv2.COLOR_BGR2RGB)
        window_size = (image_pygame.shape[0], image_pygame.shape[1])
        if first_loop:
            #for performance, don't set the mode every frame. but we don't know the aspect ratio of the webcam 
            #until we start processing it. so set_mode is in the main loop, but only used the first time around
            screen = pygame.display.set_mode(window_size, pygame.DOUBLEBUF)
            first_loop = False
        pygame.display.set_caption(window_title)
        image_surface = pygame.surfarray.make_surface(image_pygame)
        screen.blit(image_surface, (0, 0))
        update_status_bar(screen, model_list, modelmode)
        pygame.display.flip()
    pygame.quit()

config = configparser.ConfigParser()
config.read('settings.ini')

tg_key = config.get('Telegram', 'api_key', fallback='')
tg_group = config.getint('Telegram', 'group_id', fallback=0)
tg_name = config.get('Telegram', 'group_name', fallback='')
openai_key = config.get('OpenAI', 'api_key', fallback='')
openai_prompt = config.get('OpenAI', 'prompt', fallback='')
openai_model = config.get('OpenAI', 'model', fallback='')
caption_text = config.get('OpenAI', 'caption', fallback='')
window_title = config.get('Settings', 'window_title', fallback='')
custom_file = config.get('Custom', 'file', fallback='')
custom_lookup = config.get('Custom', 'lookup', fallback='')
custom_name = config.get('Custom', 'name', fallback='')
model_list = [custom_name, 'Imagenet', 'YOLO', 'ChatGPT -- Press action to post']
asyncio.run(main())