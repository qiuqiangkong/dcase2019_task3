import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import numpy as np
import argparse
import h5py
import math
import time
import logging
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utilities import (create_folder, get_filename, create_logging, 
    load_scalar, calculate_metrics)
from data_generator import DataGenerator
from models import Cnn_9layers
from losses import event_spatial_loss
from evaluate import Evaluator
from pytorch_utils import move_data_to_gpu, forward
import config


def train(args):
    '''Train. Model will be saved after several iterations. 
    
    Args: 
      dataset_dir: string, directory of dataset
      workspace: string, directory of workspace
      audio_type: 'foa' | 'mic'
      holdout_fold: '1' | '2' | '3' | '4' | 'none', where -1 indicates using 
          all data without validation for training
      model_type: string, e.g. 'Cnn_9layers'
      batch_size: int
      cuda: bool
      mini_data: bool, set True for debugging on a small part of data
    '''

    # Arugments & parameters
    dataset_dir = args.dataset_dir
    workspace = args.workspace
    audio_type = args.audio_type
    holdout_fold = args.holdout_fold
    model_type = args.model_type
    batch_size = args.batch_size
    cuda = args.cuda and torch.cuda.is_available()
    mini_data = args.mini_data
    filename = args.filename
    
    mel_bins = config.mel_bins
    frames_per_second = config.frames_per_second
    classes_num = config.classes_num
    max_validate_num = 10   # Number of audio recordings to validate
    reduce_lr = True        # Reduce learning rate after several iterations
    
    # Paths
    if mini_data:
        prefix = 'minidata_'
    else:
        prefix = ''
        
    metadata_dir = os.path.join(dataset_dir, 'metadata_dev')
        
    features_dir = os.path.join(workspace, 'features', 
        '{}{}_{}_logmel_{}frames_{}melbins'.format(prefix, audio_type, 
        'dev', frames_per_second, mel_bins))
        
    scalar_path = os.path.join(workspace, 'scalars', 
        '{}{}_{}_logmel_{}frames_{}melbins'.format(prefix, audio_type, 
        'dev', frames_per_second, mel_bins), 'scalar.h5')
        
    checkpoints_dir = os.path.join(workspace, 'checkpoints', filename, 
        '{}_{}{}_{}_logmel_{}frames_{}melbins'.format(model_type, prefix, audio_type, 
        'dev', frames_per_second, mel_bins), model_type, 
        'holdout_fold={}'.format(holdout_fold))
    create_folder(checkpoints_dir)
    
    temp_submissions_dir = os.path.join(workspace, '_temp', 'submissions', filename, 
        '{}_{}{}_{}_logmel_{}frames_{}melbins'.format(model_type, prefix, audio_type, 
        'dev', frames_per_second, mel_bins))
    create_folder(temp_submissions_dir)
    
    logs_dir = os.path.join(args.workspace, 'logs', filename, args.mode, 
        '{}_{}{}_{}_logmel_{}frames_{}melbins'.format(model_type, prefix, audio_type, 
        'dev', frames_per_second, mel_bins), 'holdout_fold={}'.format(holdout_fold))
    create_logging(logs_dir, filemode='w')
    logging.info(args)
        
    # Load scalar
    scalar = load_scalar(scalar_path)
    
    # Model
    Model = eval(model_type)
    model = Model(classes_num)
    
    if cuda:
        model.cuda()
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999),
        eps=1e-08, weight_decay=0., amsgrad=True)

    # Data generator
    data_generator = DataGenerator(
        features_dir=features_dir, 
        scalar=scalar, 
        batch_size=batch_size, 
        holdout_fold=holdout_fold)
        
    # Evaluator
    evaluator = Evaluator(
        model=model, 
        data_generator=data_generator, 
        cuda=cuda)

    train_bgn_time = time.time()
    iteration = 0

    # Train on mini batches
    for batch_data_dict in data_generator.generate_train():
                 
        # Evaluate
        if iteration % 100 == 0:

            logging.info('------------------------------------')
            logging.info('Iteration: {}'.format(iteration))

            train_fin_time = time.time()
            evaluator.evaluate(
                data_type='train', 
                metadata_dir=metadata_dir, 
                submissions_dir=temp_submissions_dir, 
                max_validate_num=max_validate_num)

            if holdout_fold != 'none':
                evaluator.evaluate(
                    data_type='validate', 
                    metadata_dir=metadata_dir, 
                    submissions_dir=temp_submissions_dir, 
                    max_validate_num=max_validate_num)

            train_time = train_fin_time - train_bgn_time
            validate_time = time.time() - train_fin_time

            logging.info(
                'Train time: {:.3f} s, validate time: {:.3f} s'
                ''.format(train_time, validate_time))

            train_bgn_time = time.time()

        # Save model
        if iteration % 1000 == 0 and iteration > 0:

            checkpoint = {
                'iteration': iteration, 
                'model': model.state_dict(), 
                'optimizer': optimizer.state_dict()}

            checkpoint_path = os.path.join(
                checkpoints_dir, '{}_iterations.pth'.format(iteration))
                
            torch.save(checkpoint, checkpoint_path)
            logging.info('Model saved to {}'.format(checkpoint_path))
            
        # Reduce learning rate
        if reduce_lr and iteration % 200 == 0 and iteration > 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.9

        # Move data to GPU
        for key in batch_data_dict.keys():
            batch_data_dict[key] = move_data_to_gpu(batch_data_dict[key], cuda)

        # Train
        model.train()
        batch_output_dict = model(batch_data_dict['feature'])
        loss = event_spatial_loss(batch_output_dict, batch_data_dict)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Stop learning
        if iteration == 5000:
            break
            
        iteration += 1


def inference_validation(args):
    '''Inference validation data. 
    
    Args: 
      dataset_dir: string, directory of dataset
      workspace: string, directory of workspace
      audio_type: 'foa' | 'mic'
      holdout_fold: '1' | '2' | '3' | '4' | 'none', where -1 indicates using 
          all data without validation for training
      model_type: string, e.g. 'Cnn_9layers'
      iteration: int
      batch_size: int
      cuda: bool
      visualize: bool
      mini_data: bool, set True for debugging on a small part of data
    '''

    # Arugments & parameters
    dataset_dir = args.dataset_dir
    workspace = args.workspace
    audio_type = args.audio_type
    holdout_fold = args.holdout_fold
    model_type = args.model_type
    iteration = args.iteration
    batch_size = args.batch_size
    cuda = args.cuda and torch.cuda.is_available()
    visualize = args.visualize
    mini_data = args.mini_data
    filename = args.filename
    
    mel_bins = config.mel_bins
    frames_per_second = config.frames_per_second
    classes_num = config.classes_num

    # Paths
    if mini_data:
        prefix = 'minidata_'
    else:
        prefix = ''
        
    metadata_dir = os.path.join(dataset_dir, 'metadata_dev')

    submissions_dir = os.path.join(workspace, 'submissions', filename, 
        '{}_{}{}_{}_logmel_{}frames_{}melbins'.format(model_type, prefix, audio_type, 
        'dev', frames_per_second, mel_bins), 'iteration={}'.format(iteration))
    create_folder(submissions_dir)
    
    logs_dir = os.path.join(args.workspace, 'logs', filename, args.mode, 
        '{}_{}{}_{}_logmel_{}frames_{}melbins'.format(model_type, prefix, audio_type, 
        'dev', frames_per_second, mel_bins), 'holdout_fold={}'.format(holdout_fold))
    create_logging(logs_dir, filemode='w')
    logging.info(args)

    # Inference and calculate metrics for a fold
    if holdout_fold != -1:
        
        features_dir = os.path.join(workspace, 'features', 
            '{}{}_{}_logmel_{}frames_{}melbins'.format(prefix, audio_type, 
            'dev', frames_per_second, mel_bins))
            
        scalar_path = os.path.join(workspace, 'scalars', 
            '{}{}_{}_logmel_{}frames_{}melbins'.format(prefix, audio_type, 
            'dev', frames_per_second, mel_bins), 'scalar.h5')
    
        checkoutpoint_path = os.path.join(workspace, 'checkpoints', filename, 
            '{}_{}{}_{}_logmel_{}frames_{}melbins'.format(model_type, '', audio_type, 
            'dev', frames_per_second, mel_bins), model_type, 
            'holdout_fold={}'.format(holdout_fold), '{}_iterations.pth'.format(iteration))
            
        # Load scalar
        scalar = load_scalar(scalar_path)
        
        # Load model    
        Model = eval(model_type)
        model = Model(classes_num)
        checkpoint = torch.load(checkoutpoint_path)
        model.load_state_dict(checkpoint['model'])
        
        if cuda:
            model.cuda()
            
        # Data generator
        data_generator = DataGenerator(
            features_dir=features_dir, 
            scalar=scalar, 
            batch_size=batch_size, 
            holdout_fold=holdout_fold)
            
        # Evaluator
        evaluator = Evaluator(
            model=model, 
            data_generator=data_generator, 
            cuda=cuda)
        
        # Calculate metrics
        data_type = 'validate'
        
        evaluator.evaluate(
            data_type=data_type, 
            metadata_dir=metadata_dir, 
            submissions_dir=submissions_dir, 
            max_validate_num=None)
        
        # Visualize reference and predicted events, elevation and azimuth
        if visualize:
            evaluator.visualize(data_type=data_type)
            
    # Calculate metrics for all 4 folds
    else:
        prediction_names = os.listdir(submissions_dir)
        prediction_paths = [os.path.join(submissions_dir, name) for \
            name in prediction_names]
        
        metrics = calculate_metrics(metadata_dir=metadata_dir, 
            prediction_paths=prediction_paths)
        
        logging.info('Metrics of {} files: '.format(len(prediction_names)))
        for key in metrics.keys():
            logging.info('    {:<20} {:.3f}'.format(key + ' :', metrics[key]))
    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example of parser. ')
    subparsers = parser.add_subparsers(dest='mode')

    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--dataset_dir', type=str, required=True)
    parser_train.add_argument('--workspace', type=str, required=True)
    parser_train.add_argument('--audio_type', type=str, choices=['foa', 'mic'], required=True)
    parser_train.add_argument('--holdout_fold', type=str, choices=['1', '2', '3', '4', 'none'], required=True, 
        help='Holdout fold. Set to -1 if using all data without validation to train. ')
    parser_train.add_argument('--model_type', type=str, required=True)
    parser_train.add_argument('--batch_size', type=int, required=True)
    parser_train.add_argument('--cuda', action='store_true', default=False)
    parser_train.add_argument('--mini_data', action='store_true', default=False)
    
    parser_inference_validation = subparsers.add_parser('inference_validation')
    parser_inference_validation.add_argument('--dataset_dir', type=str, required=True)
    parser_inference_validation.add_argument('--workspace', type=str, required=True)
    parser_inference_validation.add_argument('--audio_type', type=str, choices=['foa', 'mic'], required=True)
    parser_inference_validation.add_argument('--holdout_fold', type=str, choices=['1', '2', '3', '4', 'none'], required=True)
    parser_inference_validation.add_argument('--model_type', type=str, required=True)
    parser_inference_validation.add_argument('--iteration', type=int, required=True)
    parser_inference_validation.add_argument('--batch_size', type=int, required=True)
    parser_inference_validation.add_argument('--cuda', action='store_true', default=False)
    parser_inference_validation.add_argument('--visualize', action='store_true', default=False)
    parser_inference_validation.add_argument('--mini_data', action='store_true', default=False)
    
    args = parser.parse_args()
    args.filename = get_filename(__file__)

    if args.mode == 'train':
        train(args)

    elif args.mode == 'inference_validation':
        inference_validation(args)

    else:
        raise Exception('Error argument!')