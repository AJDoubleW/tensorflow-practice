import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs
import tensorflow as tf
import pandas
import numpy
from pprint import pprint
import matplotlib.pyplot as plt

class RetrievalModel(tfrs.models.Model):
  def __init__(self, query_model, candidate_model, task_layer):
    super().__init__()
    self.query_model = query_model
    self.candidate_model = candidate_model
    self.task_layer = task_layer

  def compute_loss(self, features, training=False):
    query_embeddings = self.query_model(features['user_id'])
    positive_candidate_embeddings = self.candidate_model(features['movie_id'])

    return self.task_layer(query_embeddings, positive_candidate_embeddings)

ratings_dataset, ratings_dataset_info = tfds.load(
  name='movielens/100k-ratings',
  split='train',
  with_info=True
)
ratings_dataset = ratings_dataset.map(lambda rating: {
  'user_id': rating['user_id'],
  'movie_id': rating['movie_id'],
  'movie_title': rating['movie_title'],
  'user_rating': rating['user_rating'],
  'timestamp': rating['timestamp']
})

trainset_size = int(0.8 * len(ratings_dataset))
tf.random.set_seed(42)
ratings_dataset_shuffled = ratings_dataset.shuffle(
  buffer_size=100_000,
  seed=42,
  reshuffle_each_iteration=False
)

ratings_trainset = ratings_dataset_shuffled.take(trainset_size)
ratings_testset = ratings_dataset_shuffled.skip(trainset_size)

timestamp_normalization_layer = tf.keras.layers.experimental.preprocessing.Normalization(axis=None)
timestamp_normalization_layer.adapt(numpy.array(list(ratings_trainset.map(lambda x: x['timestamp']))))

for rating in ratings_trainset.take(3).as_numpy_iterator():
  print(f"Raw timestamp: {rating['timestamp']} => Normalized timestamp: {timestamp_normalization_layer(rating['timestamp'])[0][0]}")

user_id_lookup_layer = tf.keras.layers.experimental.preprocessing.StringLookup(mask_token=None)
user_id_lookup_layer.adapt(ratings_trainset.map(lambda x: x['user_id']))

USER_ID_EMBEDDING_DIM = 32
user_id_embedding_layer = tf.keras.layers.Embedding(input_dim=user_id_lookup_layer.vocabulary_size(), output_dim=USER_ID_EMBEDDING_DIM)
user_id_model = tf.keras.Sequential([user_id_lookup_layer, user_id_embedding_layer])

movie_id_lookup_layer = tf.keras.layers.experimental.preprocessing.StringLookup(mask_token=None)
movie_id_lookup_layer.adapt(ratings_trainset.map(lambda x: x['movie_id']))

MOVIE_ID_EMBEDDING_DIM = USER_ID_EMBEDDING_DIM
movie_id_embedding_layer = tf.keras.layers.Embedding(input_dim=movie_id_lookup_layer.vocabulary_size(), output_dim=MOVIE_ID_EMBEDDING_DIM)
movie_id_model = tf.keras.Sequential([movie_id_lookup_layer, movie_id_embedding_layer])
print(
    f"Embedding for the movie 898:\n {movie_id_model('898')}"
)

movie_title_vectorization_layer = tf.keras.layers.experimental.preprocessing.TextVectorization()
movie_title_vectorization_layer.adapt(ratings_trainset.map(lambda x: x['movie_title']))
print(
    "Vocabulary[40:50] -> ",
    movie_title_vectorization_layer.get_vocabulary()[40:50]
)

print(
    "Vectorized title for 'Postman, The (1997)'\n",
    movie_title_vectorization_layer('Postman, The (1997)')
)

MOVIE_TITLE_EMBEDDING_DIM = MOVIE_ID_EMBEDDING_DIM
movie_title_embedding_layer = tf.keras.layers.Embedding(input_dim=len(movie_title_vectorization_layer.get_vocabulary()), output_dim=MOVIE_TITLE_EMBEDDING_DIM, mask_zero=True)
movie_title_model = tf.keras.Sequential([movie_title_vectorization_layer, movie_title_embedding_layer, tf.keras.layers.GlobalAveragePooling1D()])

query_model = user_id_model
candidate_model = movie_id_model

retrieval_ratings_trainset = ratings_trainset.map(lambda rating: {
  'user_id': rating['user_id'],
  'movie_id': rating['movie_id']
})
retrieval_ratings_testset = ratings_testset.map(lambda rating: {
  'user_id': rating['user_id'],
  'movie_id': rating['movie_id']
})

movies_dataset, movies_dataset_info = tfds.load(name='movielens/100k-movies', split='train', with_info=True)
print(tfds.as_dataframe(movies_dataset.take(5), movies_dataset_info))
candidates_corpus_dataset = movies_dataset.map(lambda movie: movie['movie_id'])

factorized_top_k_metrics = tfrs.metrics.FactorizedTopK(candidates=candidates_corpus_dataset.batch(128).map(candidate_model))
retrieval_task_layer = tfrs.tasks.Retrieval(metrics=factorized_top_k_metrics)

retrieval_cached_ratings_trainset = retrieval_ratings_trainset.shuffle(100_000).batch(8192).cache()
retrieval_cached_ratings_testset = retrieval_ratings_testset.batch(4096).cache()
NUM_EPOCHS = 5
movielens_retrieval_model = RetrievalModel(query_model, candidate_model, retrieval_task_layer)
OPTIMIZER_STEP_SIZE = 0.1
movielens_retrieval_model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=OPTIMIZER_STEP_SIZE))
history = movielens_retrieval_model.fit(
  retrieval_cached_ratings_trainset,
  validation_data=retrieval_cached_ratings_testset,
  validation_freq=1,
  epochs=NUM_EPOCHS
)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("Model loss during training")
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

plt.plot(history.history["factorized_top_k/top_100_categorical_accuracy"])
plt.plot(history.history["val_factorized_top_k/top_100_categorical_accuracy"])
plt.title("Model accuracies during training")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.legend(["train", "test"], loc="upper right")
plt.show()