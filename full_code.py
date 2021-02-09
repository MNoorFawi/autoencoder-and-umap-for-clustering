import pandas as pd

reviews = pd.read_csv("IMDB Dataset.csv")
reviews.head()

#                                               review sentiment
# 0  One of the other reviewers has mentioned that ...  positive
# 1  A wonderful little production. <br /><br />The...  positive
# 2  I thought this was a wonderful way to spend ti...  positive
# 3  Basically there's a family where a little boy ...  negative
# 4  Petter Mattei's "Love in the Time of Money" is...  positive

reviews.shape
# (50000, 2)

import re
from sklearn.feature_extraction import text

stop_words = text.ENGLISH_STOP_WORDS

def clean_review(review, stopwords):
    html_tag = re.compile('<.*?>')
    cleaned_review = re.sub(html_tag, "", review).split()
    cleaned_review = [i for i in cleaned_review if i not in stopwords]
    return " ".join(cleaned_review)

## cleaning the review column
reviews["cleaned_review"] = reviews["review"].apply(lambda x: clean_review(x, stop_words))

from sklearn.model_selection import train_test_split

## we will do the splitting using a random state to ensure same splitting every time
X_train, X_test, y_train, y_test = train_test_split(reviews.cleaned_review, 
                                                    reviews.sentiment, 
                                                    test_size = .5,
                                                    random_state = 13)
                                                    
from sklearn.feature_extraction.text import CountVectorizer

## maximum features to keep (based on frequency)
max_features = 5000

## stop words were already removed before
vectorizer = CountVectorizer(lowercase = True, stop_words = stop_words,
                             max_features = max_features)
vectorizer.fit(X_train) 
review_vectors = vectorizer.transform(X_train) 
review_vectors = review_vectors.toarray() 
print(len(vectorizer.vocabulary_))
# 5000

from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

## arbitrary number of clusters
kmeans = KMeans(n_clusters = 3, random_state = 13).fit_predict(review_vectors)
tsne = TSNE(n_components = 2, metric = "euclidean", random_state = 13).fit_transform(review_vectors)

plt.scatter(tsne[:, 0], tsne[:, 1], c = kmeans, s = 1)
plt.show()
# plt.clf() # to clear it

from keras.models import Model
from keras.layers import Dense, Input
from keras.preprocessing import sequence

max_len = 500
review_train = sequence.pad_sequences(review_vectors, maxlen = max_len)

## a subset from the test data
review_test = vectorizer.transform(X_test[:2000]).toarray()
review_test = sequence.pad_sequences(review_test, maxlen = max_len)

## define the encoder
inputs_dim = review_train.shape[1]
encoder = Input(shape = (inputs_dim, ))
e = Dense(1024, activation = "relu")(encoder)
e = Dense(512, activation = "relu")(e)
e = Dense(256, activation = "relu")(e)


## bottleneck layer
n_bottleneck = 10
## defining it with a name to extract it later
bottleneck_layer = "bottleneck_layer"
bottleneck = Dense(n_bottleneck, name = bottleneck_layer)(e)

## define the decoder (in reverse)
decoder = Dense(256, activation = "relu")(bottleneck)
decoder = Dense(512, activation = "relu")(decoder)
decoder = Dense(1024, activation = "relu")(decoder)


## output layer
output = Dense(inputs_dim)(decoder)
## model
model = Model(inputs = encoder, outputs = output)
model.summary()

## extracting the bottleneck layer we are interested in the most
bottleneck_encoded_layer = model.get_layer(name = bottleneck_layer).output
## the model to be used after training the autoencoder to refine the data
encoder = Model(inputs = model.input, outputs = bottleneck_encoded_layer)

model.compile(loss = "mse", optimizer = "adam")

history = model.fit(
    review_train,
    review_train,
    batch_size = 32,
    epochs = 25,
    verbose = 1,
    validation_data = (review_test, review_test)
)

history_dict = history.history
loss_values = history_dict["loss"]
val_loss_values = history_dict["val_loss"]
epochs = range(25)
plt.plot(epochs, loss_values, "bo", label = "training loss")
plt.plot(epochs, val_loss_values, "b", label = "validation loss")
plt.title("Training & validation loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.show()
# plt.clf() # to clear

## representing the data in lower dimensional representation or embedding
review_encoded = encoder.predict(review_train)
review_encoded.shape
# (25000, 10)

## install umap library use
# pip install umap-learn

import umap.umap_ as umap

review_umapped = umap.UMAP(n_components = n_bottleneck / 2, 
                           metric = "euclidean",
                           n_neighbors = 50, 
                           min_dist = 0.0,
                           random_state = 13).fit_transform(review_encoded)
review_umapped.shape
# (25000, 5)

from sklearn.manifold import Isomap
import numpy as np
np.random.seed(13)

review_isomapped = Isomap(n_components = n_bottleneck / 2,
                          n_neighbors = 50,
                          metric = "euclidean").fit_transform(review_encoded)
                          
from sklearn.cluster import DBSCAN
import numpy as np
np.random.seed(13)

clusters = DBSCAN(
    min_samples = 50,
    eps = 1
).fit_predict(review_umapped)
len(set(clusters))
# 5

import seaborn as sns

tsne2 = TSNE(2, metric = "euclidean", random_state = 13).fit_transform(review_encoded)

sns.scatterplot(tsne2[:, 0], tsne2[:, 1], 
                hue = clusters, palette = "deep",
                alpha = 0.9, s = 1,
                legend = "full")
                
kmeans2 = KMeans(n_clusters = 5, random_state = 13).fit_predict(review_umapped)
plt.scatter(tsne2[:, 0], tsne2[:, 1], c = kmeans2, s = 1) 
plt.show()

kmeans3 = KMeans(n_clusters = 5, random_state = 13).fit_predict(review_encoded)
plt.scatter(tsne2[:, 0], tsne2[:, 1], c = kmeans3, s = 1) 
plt.show()