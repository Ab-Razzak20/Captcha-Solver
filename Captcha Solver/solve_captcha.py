
import numpy as np
from PIL import Image
from tensorflow.python.keras.models import load_model, Model
import time
import random
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


TRAINING_DIR = 'C:/Users/raz/MyDatasets/data/'

model = None
model5 = load_model(TRAINING_DIR + "model/imitate_5_model0.h5")
model5 = load_model(TRAINING_DIR + "model/my_model.h5")

LETTERSTR = "0123456789ABCDEFGHJKLMNPQRSTUVWXYZ*"

correct, wrong = 0, 0

for i in range(1, 16):

    t1 = time.time()

    img = Image.open('test/' + str(i) + '.jpg')
    captcha = img
    captcha.convert("RGB").save('captcha.jpg', 'JPEG')

    model = model5

    prediction = model.predict(np.stack([np.array(Image.open('captcha.jpg'))/255.0]))

    total_accuracy = 1
    answer = ""
    # print(prediction)

    for predict in prediction:
        accuracy = predict[0][np.argmax(predict[0])]
        print(accuracy, end='\t')
        total_accuracy *= accuracy

        answer += LETTERSTR[np.argmax(predict[0])]
        # print()

    print('\t---  ', i, answer, total_accuracy, time.time()-t1)








