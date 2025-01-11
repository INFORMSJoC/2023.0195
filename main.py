import yaml
import argparse

import os
import logging

from utils import *
from trainer import *


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])

    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    parser.add_argument('--trainer', type=str, default='ContrastiveMultiMlpTrainer', help='The trainer to execute')
    parser.add_argument('--config', type=str, default='contrastive_multi_mlp.yml', help='Path to the config file')
    parser.add_argument('--verbose', type=str, default='info', help='Verbose level: info | debug | warning | critical')
    parser.add_argument('--doc', type=str, default='ContrastiveMultiMlp', help='A string for documentation purpose')
    parser.add_argument('--comment', type=str, default='', help='A string for experiment comment')
    parser.add_argument('--result', type=str, default='assets', help='Path for saving running related data.')
    parser.add_argument('--test', action='store_true', help='Whether to test the model')
    parser.add_argument('--resume_training', action='store_true', help='Whether to resume training')

    args = parser.parse_args()
    args.log = os.path.join(args.result, 'log', args.doc)
    args.checkpoint = os.path.join(args.result, 'checkpoint', args.doc)

    # parse config file
    with open(os.path.join('conf', args.config), 'r') as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    # setup logger
    level = getattr(logging, args.verbose.upper(), None)
    if not isinstance(level, int):
        raise ValueError('level {} not supported'.format(args.verbose))

    handler1 = logging.StreamHandler()
    if not os.path.exists(args.log):
        os.makedirs(args.log)
    handler2 = logging.FileHandler(os.path.join(args.log, 'stdout.txt'))
    formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
    handler1.setFormatter(formatter)
    handler2.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    logger.setLevel(level)

    # add device
    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    device = prepare_device()
    logging.info("Using device: {}".format(device))
    new_config.device = device

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    # torch.backends.cudnn.benchmark = True

    return args, new_config


def main():
    args, config = parse_args_and_config()
    logging.info("Writing log file to {}".format(args.log))
    logging.info("Exp instance id = {}".format(os.getpid()))
    logging.info("Exp comment = {}".format(args.comment))
    logging.info("Args = {}".format(args))
    logging.info("Config = {}".format(config))

    '''
    load trainer
    '''
    trainer = eval(args.trainer)(args, config)
    trainer.train()
    if args.test is True:
        trainer.test(load_pre_train=True)


if __name__ == '__main__':
    main()
