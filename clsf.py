from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble
from sklearn.datasets import fetch_20newsgroups

import pandas, xgboost, numpy, textblob, string
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers

# loading data set, and create a dateframe, column name:text, label
# categories = ['sci.crypt','sci.electronics','sci.med','sci.space']
categories = ['sci.crypt', 'sci.electronics']
newsgourp = fetch_20newsgroups(subset='all', categories=categories)
trainDF = pandas.DataFrame(data=numpy.c_[newsgourp['data'], newsgourp['target']], columns=['text']+['label'])

# divide into training set and verifying set, encode label for learning model
train_x, valid_x, train_y, valid_y = model_selection.train_test_split(trainDF['text'], trainDF['label'])
encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y)
valid_y = encoder.fit_transform(valid_y)

# Feature Engineering: using embedding
embeddings_index = {}
for i, line in enumerate(open('E:/Data/wiki-news-300d-1M.vec/wiki-news-300d-1M.vec',encoding='utf-8')):
    values = line.split()
    embeddings_index[values[0]] = numpy.asanyarray(values[1:], dtype='float32')

# cut off, transfer into sequence and pad them.
token = text.Tokenizer()
token.fit_on_texts(trainDF['text'])
word_index = token.word_index
train_seq_x = sequence.pad_sequences(token.texts_to_sequences(train_x), maxlen=700)
valid_seq_x = sequence.pad_sequences(token.texts_to_sequences(valid_x), maxlen=700)
embedding_matrix = numpy.zeros((len(word_index)+1, 300))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector


def train_model(classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False):
    classifier.fit(feature_vector_train, label)
    predictions = classifier.predict(feature_vector_valid)
    if is_neural_net:
        predictions = predictions.argmax(axis=-1)
    return metrics.accuracy_score(predictions, valid_y)

def create_cnn():
    input_layer = layers.Input((700,))
    embedding_layer = layers.Embedding(len(word_index)+1, 300, weights=[embedding_matrix], trainable=False)(input_layer)
    embedding_layer = layers.SpatialDropout1D(0.3)(embedding_layer)
    conv_layer = layers.Convolution1D(100, 3, activation='relu')(embedding_layer)
    # nor_layer = layers.BatchNormalization()(conv_layer)
    pooling_layer = layers.GlobalMaxPool1D()(conv_layer)
    output_layer1 = layers.Dense(50, activation='relu')(pooling_layer)
    output_layer1 = layers.Dropout(0.25)(output_layer1)
    # nor_layer2 = layers.BatchNormalization()(output_layer1)
    output_layer2 = layers.Dense(1, activation='softmax')(output_layer1)
    model = models.Model(inputs=input_layer, outputs=output_layer2)
    model.compile(optimizer=optimizers.Adam(lr=0.01), loss='binary_crossentropy')   # categorical_crossentropy
    return model

classifier = create_cnn()
accuracy = train_model(classifier, train_seq_x, train_y, valid_seq_x, is_neural_net=True)
print('CNN, Word Embeddings:',accuracy)