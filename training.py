import warnings
from matplotlib import pyplot as plt
import random
import json
import pickle
import numpy as np

from nltk.stem import WordNetLemmatizer

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD

from pythainlp import word_tokenize
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')


lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

words = []
classes = []
documents = []

# Word tokenize
for intent in intents['intents']:
    for phrases in intent['phrases']:
        word_list = word_tokenize(
            phrases, engine="deepcut", keep_whitespace=False)
        words.extend(word_list)
        documents.append((word_list, intent['value']))
        if intent['value'] not in classes:
            classes.append(intent['value'])

words = sorted(set(words))
classes = sorted(set(classes))

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

training = []
output_empty = [0] * len(classes)

# Word Embedding
for document in documents:
    bag = []
    word_phrases = document[0]
    for word in words:
        bag.append(1) if word in word_phrases else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training)

# Split data
x = list(training[:, 0])
y = list(training[:, 1])
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3)

# Create model
model = Sequential()
model.add(Dense(128, input_shape=(len(x_train[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(y_train[0]), activation='softmax'))

# Optimizer
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9)

# Compile model
model.compile(loss='categorical_crossentropy',
              optimizer=sgd, metrics=['accuracy'])

# Fit model
hist = model.fit(np.array(x), np.array(y),
                 epochs=130, batch_size=8, verbose=1, validation_data=(np.array(x_test), np.array(y_test)))
model.save('chatbotmodel.h5', hist)
model.summary()


# Evaluate the model
train_results = model.evaluate(x_train, y_train, verbose=1)
print("train loss, train acc:", train_results)

test_results = model.evaluate(x_test, y_test, verbose=1)
print("test loss, test acc:", test_results)

print(tf.__version__)

# summarize hist for accuracy
# plt.plot(hist.history['accuracy'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()

# # summarize hist for loss
# plt.plot(hist.history['loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()
