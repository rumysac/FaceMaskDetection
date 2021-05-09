from PIL import Image
import seaborn as sns
from tensorflow.math import confusion_matrix
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img ,img_to_array
from tensorflow.keras import Model
from tensorflow.keras.layers import  Flatten, Dense, Dropout
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras import optimizers
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os

DIRECTORY1 = r"/Users/rumeysacelik/Desktop/FaceMaskDetection2/FaceMaskDataset/Train/"
DIRECTORY2 = r"/Users/rumeysacelik/Desktop/FaceMaskDetection2/FaceMaskDataset/Validation/"
DIRECTORY3 = r"/Users/rumeysacelik/Desktop/FaceMaskDetection2/FaceMaskDataset/Test"
CATEGORIES = ["WithMask", "WithoutMask"]

# grab the list of images in our dataset directory, then initialize
# the list of data (i.e., images) and class images
print("[INFO] loading images...")

data = []
labels = []

for category in CATEGORIES:
    path = os.path.join(DIRECTORY1,DIRECTORY2,DIRECTORY3, category)
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        image = load_img(img_path, target_size=(128, 128))
        image = img_to_array(image)
        image = preprocess_input(image)

        data.append(image)
        labels.append(category)

# perform one-hot encoding on the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

data = np.array(data, dtype="float32")
labels = np.array(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels,
    test_size=0.20, stratify=labels, random_state=42)
    
"""## Data Augmentation"""

target_size=(128,128)
batch_size = 64
EPOCHS = 8
CLASSES = ['Without Mask ','With Mask']

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.25,
    horizontal_flip=True,
    vertical_flip=True)

train = train_datagen.flow_from_directory(
    DIRECTORY1,
    target_size=target_size,
    batch_size=batch_size,
    color_mode='rgb',    
    shuffle=True,
    seed=42,
    class_mode='categorical')

valid_datagen = ImageDataGenerator(rescale=1./255)

valid = valid_datagen.flow_from_directory(
    DIRECTORY2,
    target_size=target_size,
    batch_size=batch_size,
    color_mode='rgb',
    shuffle=False,    
    class_mode='categorical')

test_datagen = ImageDataGenerator(rescale=1./255)

test = test_datagen.flow_from_directory(
    DIRECTORY3,
    target_size=target_size,
    batch_size=batch_size,
    color_mode='rgb', 
    shuffle=False,    
    class_mode='categorical')
    
train_set = train_datagen.flow_from_directory(DIRECTORY1, batch_size=batch_size, class_mode='categorical', target_size=target_size)
validation_set = test_datagen.flow_from_directory(DIRECTORY2,target_size=target_size)

"""## Build Model"""

def craete_model():
    
    denseNet_model = DenseNet201(input_shape=target_size + (3,), weights='imagenet', include_top=False)
    denseNet_model.trainable = False
    
    flatten = Flatten()(denseNet_model.layers[-1].output)
    fc = Dense(units=512, activation='relu')(flatten)
    dropout = Dropout(0.35)(fc)
    output = Dense(2, activation='softmax')(dropout)
   
    model = Model(inputs=denseNet_model.input, outputs=output)
    
    model.summary()
    
    return model


model = craete_model()


"""## Train Model"""

starter_learning_rate = 1e-2
end_learning_rate = 1e-6
decay_steps = 10000
learning_rate = optimizers.schedules.PolynomialDecay(starter_learning_rate,decay_steps,end_learning_rate,power=0.4)

# Define Optimizer, Loss & Metrics


opt = optimizers.Adam(learning_rate=learning_rate)
loss = CategoricalCrossentropy()
met = 'accuracy'

# Compile the Model
model.compile(optimizer=opt, loss=loss, metrics=[met])

my_callbacks = [
                EarlyStopping(monitor='val_accuracy', min_delta=1e-5, patience=5, mode='auto',restore_best_weights=False, verbose=1),
                ModelCheckpoint(filepath='mymodel.h5', monitor='accuracy', save_best_only=True, save_weights_only=False, mode='auto', save_freq='epoch', verbose=1)
]

history = model.fit(train_set,
                    epochs=EPOCHS, steps_per_epoch=len(train_set), # How many mini_batchs we have inside each epoch.
                    validation_data=validation_set,
                    callbacks=[my_callbacks],
                    verbose=1)

print('\n*** Fit is over ***')
model.save('mymodel.h5')
#model.save_weights("my_model.h5")

train_loss = np.array(history.history['loss'])
val_loss = np.array(history.history['val_loss'])
plt.semilogy(train_loss, label='Train Loss')
plt.semilogy(val_loss, label='Validation Loss')


plt.legend(loc='upper right')
plt.xlabel('Epoch')
plt.ylabel('Loss - Cross Entropy')
plt.title('Train and Validation Loss')


plt.show()

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend(loc='lower right')
plt.xlabel('Epoch'),
plt.ylabel('Accuracy')
plt.title('Train and Validation Accuracy')
plt.show()

test_set = test_datagen.flow_from_directory(DIRECTORY3,target_size,shuffle=False)

# Model Evaluate
loss, accuracy = model.evaluate(test_set)
print('Test Accuracy: ', '\033[1m',round(accuracy*100, 2),'%\033[0m')

# True Label & Predict of a particular Batch
image, label = test_set.next()
num_imgs = 20
lab_names = ['With Mask','Without Mask ']
images = image[0:num_imgs,:,:,:]
labels = label[0:num_imgs,:]
predict = np.round(model.predict(images))

image_rows = 4
image_col = int(num_imgs/image_rows)

_, axs = plt.subplots(image_rows, image_col, figsize=(32,8))
axs = axs.flatten()

for i in range(num_imgs):
    img = images[i,:,:,:]
    lab = labels[i,:]
    axs[i].imshow(img)
    pred = predict[i]
    axs[i].axis('off')
    lab, pred = np.argmax(lab), np.argmax(pred)
    axs[i].set_title(label = f'y: {lab_names[lab]}  |  y_pred: {lab_names[pred]}', fontsize=14)

plt.show()

y_pred = model.predict(test_set).argmax(axis=-1)
y_test = test_set.classes

Confusion_Matrix = confusion_matrix(y_test,y_pred)

fig, ax = plt.subplots(figsize=(25,10))
sns.heatmap(Confusion_Matrix,xticklabels=CLASSES,yticklabels=CLASSES, ax=ax, annot=True,fmt="1.0f",cbar=False,annot_kws={"size": 40})
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
plt.title("Confusion matrix",fontsize=30)

model.save('mymodel.h5')
