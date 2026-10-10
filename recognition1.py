import cv2
import mediapipe as mp
import pandas as pd  
import os
import numpy as np
import pickle
import keyboard
from PIL import Image, ImageDraw, ImageFont
import telepot

# Telegram bot credentials are read from the environment (never hardcode them)
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')




names = {'ka':'ಕ','kha':'ಖ','ga':'ಗ','gha':'ಘ','nga':'ಙ','ca':'ಚ','cha':'ಛ','ja':'ಜ','jha':'ಝ',
          'nya':'ಞ','ta':'ಟ','taa':'ಠ','da':'ಡ','daa':'ಢ','na':'ಣ','tha':'ತ','thaa':'ಥ','dha':'ದ',
           'dhaa':'ಧ','nha':'ನ','pa':'ಪ','paa':'ಫ','ba':'ಬ','baa':'ಭ','ma':'ಮ','ya':'ಯ','ra':'ರ',
         'lha':'ಲ','va':'ವ','she':'ಶ','sha':'ಷ','sa':'ಸ','ha':'ಹ','la':'ಳ','a':'ಅ','aa':'ಆ','i':'ಇ',
         'ii':'ಈ','u':'ಉ','uu':'ಊ','ri':'ಋ','e':'ಎ','ee':'ಏ','ai':'ಐ','o':'ಒ','oo':'ಓ','au':'ಔ','ao':'ಅಂ',
         'aha':'ಅಃ', 'Namaskara':'ನಮಸ್ಕಾರ', 'Appa':'ಅಪ್ಪ','Amma':'ಅಮ್ಮ','Aytha':'ಆಯ್ತಾ','Shuba':'ಶುಭ','Munjane':'ಮುಂಜಾನೆ',
         'Madhyana':' ಮಧ್ಯಾನ','Ratri':'ರಾತ್ರಿ','yenmadutidhiya':'ಏನು ಮಾಡುತ್ತಿದಿಯ','Thindi':'ತಿಂಡಿ','Dhanyavadagalu':' ಧನ್ಯವಾದಗಳು',
         'OhHwdha':' ಓ‌ ಹೌದ','prayathna':'ಪ್ರಯತ್ನ','Vandhanegalu':'ವಂದನೆಗಳು','varsha':'ವರ್ಷ','shreyas':'ಶ್ರೇಯಸ್','kelas':'ಕೆಲಸ',
         'holleyadagali':'ಒಳ್ಳೆಯದಾಗಲಿ','Results':'ಫಲಿತಾಂಶ','Engineering':'ಇಂಜಿನಿಯರಿಂಗ್','Students':'ವಿದ್ಯಾರ್ಥಿಗಳು','Teachers':'ಶಿಕ್ಷಕರು',
         'yegidira':' ಹೇಗಿದೀರ','trust':'ನಂಬಿಕೆ','APS':' ಎ‌ಪಿಎಸ್','College':' ಕಾಲೇಜು','Rajashree':'ರಾಜಶ್ರೀ','Bhagyashree':'ಭಾಗ್ಯಶ್ರೀ',
         'Intelligent':'ಬುದ್ಧಿವಂತ','HmmSari':'ಹಾಂ ಸಾರಿ','Goal':'ಗುರಿ','Understand':'ಅರ್ಥಮಾಡಿಕೊಳ್ಳಿ','Sameerana':' ಸಮೀರನಾ',
         'Principal':'ಪ್ರಾಂಶುಪಾಲರು','VTU':'ವಿ‌ಟಿಯು','Parents':'ಪೋಷಕರು','yellaru':'ಯಾಲ್ಲರು','Rajathamahotsava':' ರಜತ ಮಹೋತ್ಸವ',
         'Vikas':'ವಿಕಾಸ್','Blessings':'ಆಶೀರ್ವಾದ'}

# load model
with open('model.pkl', 'rb') as f:
    svm = pickle.load(f)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

Word = ''

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    image = cv2.flip(image, 1)
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
        
##        print(hand_landmarks)
        
        data = str(hand_landmarks)
        data = data.strip().split('\n')

        garbage = ['landmark {', '  visibility: 0.0', '  presence: 0.0', '}']

        without_garbage = []

        for i in data:
            if i not in garbage:
                without_garbage.append(i)
                        
        clean = []

        for i in without_garbage:
            i = i.strip()
            clean.append(i[2:])

        for i in range(0, len(clean)):
            clean[i] = float(clean[i])
        
##        print(clean)

        clean = np.array(clean)
        Class = svm.predict(clean.reshape(-1,63))
        Class = Class[0]
        Class=names[Class]
        print(Class)
        if keyboard.is_pressed('a'):
              Word += Class
        if keyboard.is_pressed('b'):
              Word += ' '
        #cv2.putText(image, str(Class), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                    #2, (255, 0, 0), 2)

        image = Image.fromarray(image)
        draw = ImageDraw.Draw(image)

        # Define the font for the text
        font = ImageFont.truetype("kannada.ttf", size=50)

       # Write the text on the image
        draw.text((100, 100),Class , fill=(0, 255, 0), font=font)

        image = np.asarray(image)

    if keyboard.is_pressed('e'):
              print(Word)
              if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
                  bot=telepot.Bot(TELEGRAM_BOT_TOKEN)
                  bot.sendMessage(TELEGRAM_CHAT_ID,str(Word))
              else:
                  print("Telegram not configured; set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID to send messages.")
              image = Image.fromarray(image)
              draw = ImageDraw.Draw(image)

              # Define the font for the text
              font = ImageFont.truetype("kannada.ttf", size=50)

             # Write the text on the image
              draw.text((50, 50),Word , fill=(0, 255, 0), font=font)

              image = np.asarray(image)
              #cv2.putText(image, str(Word), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                    #2, (255, 0, 0), 2)
              Word = ''
              
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(1000) & 0xFF == ord('q'):
      break
              
cap.release()
cv2.destroyAllWindows()
