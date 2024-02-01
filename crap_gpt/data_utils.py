import numpy as np
import pandas as pd
import tensorflow as tf
from biom import load_table #Need to ask Kalen for biom

def get_sequencing_dataset(table_path):
    if type(table_path) == str:
        table = load_table(table_path)
    else:
        table = table_path
    o_ids = tf.constant(table.ids(axis='observation'))
    table = table.transpose()
    data = table.matrix_data.tocoo()
    row_ind = data.row
    col_ind = data.col
    values = data.data
    indices = [[r, c] for r, c in zip(row_ind, col_ind)]
    table_data = tf.sparse.SparseTensor(indices=indices, values=values,dense_shape=table.shape)
    table_data = tf.sparse.reorder(table_data)
    get_asv_id = lambda x: tf.gather(o_ids, x.indices)
    return (tf.data.Dataset.from_tensor_slices(table_data)
                           .map(get_asv_id, num_parallel_calls=tf.data.AUTOTUNE)
                           .prefetch(tf.data.AUTOTUNE)
)

def combine_seq_diet_dataset(seq_dataset, diet_dataset, batch_size):
    sequence_tokenizer = tf.keras.layers.TextVectorization(
        max_tokens=7,
        split='character',
        output_mode='int',
        output_sequence_length=100)
    sequence_tokenizer.adapt(seq_dataset.take(1))

    seq_dataset = seq_dataset.map(lambda x: sequence_tokenizer(x))
    dataset_size = seq_dataset.cardinality()
    return (tf.data.Dataset
            .zip(tf.data.Dataset.range(dataset_size), seq_dataset, diet_dataset)
            .shuffle(dataset_size, reshuffle_each_iteration=False)
            .prefetch(tf.data.AUTOTUNE)
    ), seq_dataset

def batch_dataset(dataset, batch_size, shuffle=False, repeat=None):
    dataset = dataset.cache()
    size = dataset.cardinality()

    if shuffle:
        dataset = dataset.shuffle(size, reshuffle_each_iteration=True)

    def pair_seq_diet(ind, seq, diet):
        return (seq, diet)

    dataset = (dataset
        .padded_batch(batch_size, padded_shapes=([], [None,100], [None]), drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
        .map(pair_seq_ti_diet)
        .prefetch(tf.data.AUTOTUNE)
    )

    if not shuffle:
        dataset = dataset.cache()
    else:
        dataset = dataset.repeat(repeat)

    return dataset.prefetch(tf.data.AUTOTUNE)
