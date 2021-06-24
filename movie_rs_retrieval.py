import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs
import tensorflow as tf
import numpy
import tempfile
import os

class MovielensModel(tfrs.Model):
  def __init__(self, user_model, movie_model, task):
    super().__init__()
    self.movie_model: tf.keras.Model = movie_model
    self.user_model: tf.keras.Model = user_model
    self.task = tf.keras.layers.Layer = task

  def compute_loss(self, features, training=False) -> tf.Tensor:
    user_embeddings = self.user_model(features['user_id'])
    positive_movie_embeddings = self.movie_model(features['movie_title'])

    return self.task(user_embeddings, positive_movie_embeddings)

ratings = tfds.load('movielens/100k-ratings', split="train")
movies = tfds.load('movielens/100k-movies', split='train')
ratings = ratings.map(lambda x: {
  'movie_title': x['movie_title'],
  'user_id': x['user_id'],
})
movies = movies.map(lambda x: x['movie_title'])

print(movies)

RANDOM_SEED = 42
TAKE_AMOUNT = 80_000

tf.random.set_seed(RANDOM_SEED)
shuffled = ratings.shuffle(100_000, seed=RANDOM_SEED, reshuffle_each_iteration=False)
train = shuffled.take(TAKE_AMOUNT)
test = shuffled.skip(TAKE_AMOUNT).take(20_000)


unique_movie_titles = numpy.unique(numpy.concatenate((list(movies.batch(1_000)))))
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

metrics = tfrs.metrics.FactorizedTopK(candidates=movies.batch(128).map(movie_model))
task = tfrs.tasks.Retrieval(metrics=metrics)

cached_train = train.shuffle(100_000).batch(8192).cache()
cached_test = test.batch(4096).cache()

model = MovielensModel(user_model, movie_model, task)
model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))
model.fit(cached_train, epochs=3)
model.evaluate(cached_test, return_dict=True)

scann_index = tfrs.layers.factorized_top_k.ScaNN(model.user_model)
scann_index.index(movies.batch(100).map(model.movie_model), movies)
_, titles = scann_index(tf.constant(['42']))

with tempfile.TemporaryDirectory() as tmp:
  path = os.path.join(tmp, 'model')
  scann_index.save(
    path,
    options=tf.saved_model.SaveOptions(namespace_whitelist=['Scann'])
  )
  loaded = tf.keras.models.load_model(path)
  scores, titles = loaded(['42'])
  print(f"Recommendations for user 42: {titles[0, :3]}")