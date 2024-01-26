import click
import crap_gpt._parameter_descriptions as desc

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

