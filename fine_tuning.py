import click
import crap_gpt._parameter_descriptions as desc
import tensorflow as tf
from crap_gpt.data_utils import (
    get_sequencing_dataset,
    combine_seq_diet_dataset,
    batch_dataset
)
import pandas as pd
import numpy as np

# Allow using -h to show help information
# https://click.palletsprojects.com/en/7.x/documentation/#help-parameter-customization
CTXSETS = {"help_option_names": ["-h", "--help"]}

@click.command()
@click.option(
    '--config-json',
    required=True,
    type=click.Path(exists=True),
    help=desc.CONFIG_JSON
)
@click.option(
    '-c', '--continue-training',
    required=False, default=False, is_flag=True,
    help=desc.CONTINUE_TRAINING
)
@click.option(
    '--output-model-summary',
    required=False, default=False, is_flag=True,
    help=desc.OUTPUT_MODEL_SUMMARY
)
def fine_tuning(config_json, continue_training, output_model_summary):
    pass


metadata = pd.read_csv('agp-metadata.txt', sep='\t', dtype=str)

print(pd.to_numeric(metadata['age_years'], errors='coerce').describe())
# batch_size=2
# training_percent=0.7
# seq_dataset = get_sequencing_dataset(table_path)
# diet_dataset = # function to extract y labels from metadata 
# dataset, _ = combine_seq_diet_dataset(seq_dataset, diet_dataset, batch_size)

# size = seq_dataset.cardinality().numpy()
# train_size = int(size*training_percent/batch_size)*batch_size

# training_dataset = dataset.take(train_size).prefetch(tf.data.AUTOTUNE)
# training_dataset = batch_dataset(training_dataset, shuffle=True)

# val_data = dataset.skip(train_size).prefetch(tf.data.AUTOTUNE)
# validation_dataset = batch_dataset(val_data)

# # print top 5 items of training_dataset
# for x, y in training_dataset.take(5):
#     print(x, y)