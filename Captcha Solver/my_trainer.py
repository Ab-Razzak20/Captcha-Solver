from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from PIL import Image
import numpy as np
import csv


LETTERSTR = "0123456789ABCDEFGHJKLMNPQRSTUVWXYZ*"
CHARLEN = 6

TRAINING_DIR = 'C:/Users/raz/MyDatasets/data/'



def toonehot(text):
    labellist = []
    for letter in text:
        onehot = [0 for _ in range(len(LETTERSTR))]
        num = LETTERSTR.find(letter)
        onehot[num] = 1
        labellist.append(onehot)
    return labellist


# Create CNN Model
print("Creating CNN model...")
inp = Input((60, 200, 3))
out = inp
out = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(out)
out = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(out)
out = BatchNormalization()(out)
out = MaxPooling2D(pool_size=(2, 2))(out)
out = Dropout(0.3)(out)
out = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(out)
out = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(out)
out = BatchNormalization()(out)
out = MaxPooling2D(pool_size=(2, 2))(out)
out = Dropout(0.3)(out)
out = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(out)
out = Conv2D(filters=128, kernel_size=(3, 3), activation='relu')(out)
out = BatchNormalization()(out)
out = MaxPooling2D(pool_size=(2, 2))(out)
out = Dropout(0.3)(out)
out = Conv2D(filters=256, kernel_size=(3, 3), activation='relu')(out)
out = BatchNormalization()(out)
out = MaxPooling2D(pool_size=(2, 2))(out)
out = Flatten()(out)
out = Dropout(0.3)(out)
out = [Dense(len(LETTERSTR), name='digit1', activation='softmax')(out),
    Dense(len(LETTERSTR), name='digit2', activation='softmax')(out),
    Dense(len(LETTERSTR), name='digit3', activation='softmax')(out),
    Dense(len(LETTERSTR), name='digit4', activation='softmax')(out),
    Dense(len(LETTERSTR), name='digit5', activation='softmax')(out),
    Dense(len(LETTERSTR), name='digit6', activation='softmax')(out)
]
model = Model(inputs=inp, outputs=out)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

print("Reading training data...")
traincsv = open(TRAINING_DIR + 'my_train_set/captcha_train.csv', 'r', encoding = 'utf8')
train_data = np.stack([np.array(Image.open(TRAINING_DIR + "my_train_set/" + row[0] + ".jpg"))/255.0 for row in csv.reader(traincsv)])
traincsv = open(TRAINING_DIR + 'my_train_set/captcha_train.csv', 'r', encoding = 'utf8')
read_label = [toonehot(row[1]) for row in csv.reader(traincsv)]
train_label = [[] for _ in range(CHARLEN)]
for arr in read_label:
    for index in range(CHARLEN):
        train_label[index].append(arr[index])
train_label = [arr for arr in np.asarray(train_label)]
print("Shape of train data:", train_data.shape)

print("Reading validation data...")
valicsv = open(TRAINING_DIR + 'my_vali_set/captcha_vali.csv', 'r', encoding = 'utf8')
vali_data = np.stack([np.array(Image.open(TRAINING_DIR + "my_vali_set/" + row[0] + ".jpg"))/255.0 for row in csv.reader(valicsv)])
valicsv = open(TRAINING_DIR + 'my_vali_set/captcha_vali.csv', 'r', encoding = 'utf8')
read_label = [toonehot(row[1]) for row in csv.reader(valicsv)]
vali_label = [[] for _ in range(CHARLEN)]
for arr in read_label:
    for index in range(CHARLEN):
        vali_label[index].append(arr[index])
vali_label = [arr for arr in np.asarray(vali_label)]
print("Shape of validation data:", vali_data.shape)

filepath=TRAINING_DIR + "model/my_model.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_digit3_accuracy', verbose=1, save_best_only=True, mode='max')
earlystop = EarlyStopping(monitor='val_digit3_accuracy', patience=40, verbose=1, mode='auto')
tensorBoard = TensorBoard(log_dir = TRAINING_DIR + "logs", histogram_freq = 1)
callbacks_list = [checkpoint, earlystop, tensorBoard]
model.fit(train_data, train_label, batch_size=20, epochs=20, verbose=2, validation_data=(vali_data, vali_label), callbacks=callbacks_list)

model.save('saved_model.h5')

