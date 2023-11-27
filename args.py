import yaml
import argparse
import os

def arg_parse():
    parser = argparse.ArgumentParser(description='Train a random forest model for drug discovery')
    parser.add_argument('--config', dest='config', type=str, default='./config/config.yaml', help='Path to the config file')
    args = parser.parse_args()
    return args

def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

