import os
import pprint
import tempfile
import numpy
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs

class RankingModel(tf.keras.Model):
  def __init__(self, user_model, movie_model, ratings):
    super().__init__()
    self.user_embeddings = user_model
    self.movie_embeddings = movie_model
    self.ratings = ratings

  def call(self, inputs):
    user_id, movie_title = inputs
    user_embedding = self.user_embeddings(user_id)
    movie_embedding = self.movie_embeddings(movie_title)

    return self.ratings(tf.concat([user_embedding, movie_embedding], axis=1))

ratings = tfds.load('movielens/100k-ratings', split='train')
ratings = ratings.map(lambda x: {
  'movie_title': x['movie_title'],
  'user_id': x['user_id'],
  'user_rating': x['user_rating']
})

class MovielensModel(tfrs.models.Model):
  def __init__(self, user_model, movie_model, ratings):
    super().__init__()
    self.ranking_model = RankingModel(user_model, movie_model, ratings)
    self.task = tfrs.tasks.Ranking(
      loss=tf.keras.losses.MeanSquaredError(),
      metrics=[tf.keras.metrics.RootMeanSquaredError()]
    )

  def compute_loss(self, features, training=False):
    rating_predictions = self.ranking_model((features['user_id'], features['movie_title']))

    return self.task(labels=features['user_rating'], predictions=rating_predictions)

RANDOM_SEED = 42
TAKE_AMOUNT = 80_000

tf.random.set_seed(RANDOM_SEED)
shuffled = ratings.shuffle(100_000, seed=RANDOM_SEED, reshuffle_each_iteration=False)
train = shuffled.take(TAKE_AMOUNT)
test = shuffled.skip(TAKE_AMOUNT).take(20_000)

unique_movie_titles = numpy.unique(numpy.concatenate((list(ratings.batch(1_000_000).map(lambda x: x['movie_title'])))))
unique_user_ids = numpy.unique(numpy.concatenate(list(ratings.batch(1_000_000).map(lambda x: x['user_id']))))

EMBEDDING_DIMENSION = 32

user_model = tf.keras.Sequential([
  tf.keras.layers.experimental.preprocessing.StringLookup(vocabulary=unique_user_ids, mask_token=None),
  tf.keras.layers.Embedding(len(unique_user_ids) + 1, EMBEDDING_DIMENSION)
])

movie_model = tf.keras.Sequential([
  tf.keras.layers.experimental.preprocessing.StringLookup(vocabulary=unique_movie_titles, mask_token=None),
  tf.keras.layers.Embedding(len(unique_movie_titles) + 1, EMBEDDING_DIMENSION)
])

ratings = tf.keras.Sequential([
  tf.keras.layers.Dense(256, activation='relu'),
  tf.keras.layers.Dense(64, activation='relu'),
  tf.keras.layers.Dense(1)
])
# print(RankingModel(user_model, movie_model, ratings)((["42"], ["One Flew Over the Cuckoo's Nest (1975)"])))

model = MovielensModel(user_model, movie_model, ratings)
model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))

cached_train = train.shuffle(100_000).batch(8192).cache()
cached_test = test.batch(4096).cache()

model.fit(cached_train, epochs=3)
model.evaluate(cached_test, return_dict=True)
