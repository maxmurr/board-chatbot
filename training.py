import random
import json
import pickle
import numpy as np

from nltk.stem import WordNetLemmatizer

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD

from pythainlp import word_tokenize

import warnings
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
train_x = list(training[:, 0])
train_y = list(training[:, 1])

# Create model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Optimizer
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9)

# Compile model
model.compile(loss='categorical_crossentropy',
              optimizer=sgd, metrics=['accuracy'])

# Fit model
hist = model.fit(np.array(train_x), np.array(train_y),
                 epochs=200, batch_size=5, verbose=1)
model.save('chatbotmodel.h5', hist)
model.summary()

# Evaluate the model
model.evaluate(train_x, train_y, verbose=1)
