# -*- coding: utf-8 -*-
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""run.py"""
import os, sys, pathlib
import click
import yaml
from model.utils.config_gpu_memory_lim import allow_gpu_memory_growth
from model.trainer import trainer
from model.generate import generate_fingerprint


def load_config(config_fname):
    config_filepath = './config/' + config_fname + '.yaml'
    if os.path.exists(config_filepath):
        print(f'cli: Configuration from {config_filepath}')
    else:
        sys.exit(f'cli: ERROR! Configuration file {config_filepath} is missing!!')
             
    with open(config_filepath, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg


def update_config(cfg, key1: str, key2: str, val):
    cfg[key1][key2] = val
    return cfg


def print_config(cfg):
    os.system("")
    print('\033[36m' + yaml.dump(cfg, indent=4, width=120, sort_keys=False) +
          '\033[0m')  
    return


@click.group()
def cli():
    pass


"""Train."""
@cli.command()
@click.argument('checkpoint_name', required=True)
@click.option('--config', '-c', default='default', type=click.STRING,
              help="Name of model configuration located in './config/.'")
@click.option('--max_epoch', default=None, type=click.INT, help='Max epoch.')
def train(checkpoint_name, config, max_epoch):
    """Train a neural audio fingerprinter.

    ex) python run.py train my_experiment --max_epoch=100
    
    If './LOG_ROOT_DIR/checkpoint/CHECKPOINT_NAME already exists, the training will resume from the latest checkpoint in the directory. 
    """
    cfg = load_config(config)
    if max_epoch: update_config(cfg, 'TRAIN', 'MAX_EPOCH', max_epoch)
    print_config(cfg)
    allow_gpu_memory_growth()
    trainer(cfg, checkpoint_name)

    
"""Generate fingerprint after training."""
@cli.command()
@click.argument('checkpoint_name', required=True)
@click.argument('checkpoint_index', required=False)
@click.option('--config', '-c', default='default', required=False,
              type=click.STRING,
              help="Name of the model configuration file located in 'config/'. Default is 'default'")
@click.option('--source', '-s', default=None, type=pathlib.Path, required=False,
              help="Custom source root directory. Use '*.wav' files in recurse subdirectories as the source.")
@click.option('--output', '-o', default=None, type=pathlib.Path, required=False,
              help="Root directory where the generated embeddings (uncompressed) will be stored. Default is OUTPUT_ROOT_DIR/CHECKPOINT_NAME defined in config.")
@click.option('--skip-dummy', default=False, is_flag=True,
              help='Exclude dummy-DB from the default source.')
def generate(checkpoint_name, checkpoint_index, config, source, output, skip_dummy):
    """Generate fingerprints from a saved checkpoint.
    
    - If CHECKPOINT_INDEX is not specified, the latest checkpoint in the OUTPUT_ROOT_DIR will be loaded.
    - The default value for the fingerprinting source is TEST_DUMMY_DB and TEST_QUERY_DB specified in config. To use a custom source, use the --source option.
    """
    print('generate!')
    print(f'train! {checkpoint_name}, {checkpoint_index}, {config}')
    print(type(checkpoint_index))
    cfg = load_config(config)
    allow_gpu_memory_growth()
    print(type(source))
    generate_fingerprint(cfg, checkpoint_name, checkpoint_index, source, output, skip_dummy)


if __name__ == '__main__':
    cli()
 