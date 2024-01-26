import tensorflow as tf
import tensorflow_datasets as tfds


# Load the dataset
(ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],                        # 'train' and 'test' indicate that you want to load both the training
                                                    #  and test subsets of the MNIST dataset

    shuffle_files=True,                             #  If dataset is split into multiple files, shuffle the order of the files. 
                                                    #  Useful for large datasets to ensure that the model does not learn the order of the data

    as_supervised=True,                             #  Dataset is returned in tuple format ((input, label))

    with_info=True,                                 #  Return the metadata of the dataset (version of dataset, # of samples, feature descriptions)
)

# ----------------------------------------------------------

# Build a traning pipeline
def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255., label                  

ds_train = ds_train.map(                                            
    normalize_img, num_parallel_calls=tf.data.AUTOTUNE)        
ds_train = ds_train.cache()                                         # 
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)   #  Shuffle to ensure randomness
ds_train = ds_train.batch(128)
ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

# ----------------------------------------------------------

# Build an evaluation pipeline
ds_test = ds_test.map(
    normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_test = ds_test.batch(128)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

#Create and train model
model = tf.keras.models.Sequential([                           # Sequential model in Keras is a linear stack of layers. Simple and straightforward
                                                               # way of building a model in Keras. Just add layers in order you want them to be connected
  tf.keras.layers.Flatten(input_shape=(28, 28)),    
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10)
])
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),                                      # Adam is an optimization algorithm that can handle sparse gradients on noisy problems
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),           # The loss function; from_logits=True indicates that the output values
                                                                                    # of the model are not normalized
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],                         # Metric used to evaluate the model is "SparseCategoricalAccuracy", which calculates
                                                                                    # how often the model's predictions match the labels
)
# Model fitting
model.fit(
    ds_train,                   # The training dataset
    epochs=6,                   # The number of times the training data is passed through the model. 
                                # An epoch = one complete presentation of the data set to be learned to a learning machine
    validation_data=ds_test,    # The test dataset used for validation. After each epoch, the model is evaluated on this dataset
)

# ----------------------------------------------------------

# import click
# import crap_gpt._parameter_descriptions as desc

# # Allow using -h to show help information
# # https://click.palletsprojects.com/en/7.x/documentation/#help-parameter-customization
# CTXSETS = {"help_option_names": ["-h", "--help"]}

# @click.command()
# @click.option(
#     required=True,
#     type=click.Path(exists=True),
#     help=desc.CONFIG_JSON
# )
# @click.option(
#     '-c', '--continue-training',
#     required=False, default=False, is_flag=True,
#     help=desc.CONTINUE_TRAINING
# )
# @click.option(
#     '--output-model-summary',
#     required=False, default=False, is_flag=True,
#     help=desc.OUTPUT_MODEL_SUMMARY
# )
# def fine_tuning(config_json, continue_training, output_model_summary):
#     pass

