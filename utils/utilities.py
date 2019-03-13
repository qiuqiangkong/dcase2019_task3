import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../evaluation_tools'))
import numpy as np
import soundfile
import librosa
import h5py
from sklearn import metrics
import logging
import matplotlib.pyplot as plt

import evaluation_metrics
import cls_feature_class
import config


def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)


def get_filename(path):
    path = os.path.realpath(path)
    name_ext = path.split('/')[-1]
    name = os.path.splitext(name_ext)[0]
    return name


def create_logging(log_dir, filemode):

    create_folder(log_dir)
    i1 = 0

    while os.path.isfile(os.path.join(log_dir, '{:04d}.log'.format(i1))):
        i1 += 1
        
    log_path = os.path.join(log_dir, '{:04d}.log'.format(i1))
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
        datefmt='%a, %d %b %Y %H:%M:%S',
        filename=log_path,
        filemode=filemode)

    # Print to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    
    return logging


def read_multichannel_audio(audio_path, target_fs=None):

    (multichannel_audio, fs) = soundfile.read(audio_path)     # (samples, channels_num)
    
    if target_fs is not None and fs != target_fs:
        (samples, channels_num) = multichannel_audio.shape
        
        multichannel_audio = np.array(
            [librosa.resample(
                multichannel_audio[:, i], 
                orig_sr=fs, 
                target_sr=target_fs) 
            for i in range(channels_num)]).T
        '''(samples, channels_num)'''

    return multichannel_audio, fs


def calculate_scalar_of_tensor(x):

    if x.ndim == 2:
        axis = 0
    elif x.ndim == 3:
        axis = (0, 1)

    mean = np.mean(x, axis=axis)
    std = np.std(x, axis=axis)

    return mean, std


def load_scalar(scalar_path):
    with h5py.File(scalar_path, 'r') as hf:
        mean = hf['mean'][:]
        std = hf['std'][:]
        
    scalar = {'mean': mean, 'std': std}
    return scalar
    
    
def scale(x, mean, std):
    return (x - mean) / std

        
def write_submission(list_dict, submissions_dir):
    '''Write predicted result to submission csv files. 
    
    Args:
      list_dict: list of dict, containing predicted event, elevation and azimuth
      submissions_dir: string
    '''
    
    frames_per_second = config.frames_per_second
    submission_frames_per_second = config.submission_frames_per_second
    
    for dict in list_dict:
        filename = '{}.csv'.format(dict['name'])
        filepath = os.path.join(submissions_dir, filename)
        
        event_matrix = dict['output_event'][0]
        elevation_matrix = dict['output_elevation'][0]
        azimuth_matrix = dict['output_azimuth'][0]
        
        # Resample predicted frames to submission format
        ratio = submission_frames_per_second / float(frames_per_second)
        resampled_event_matrix = resample_matrix(event_matrix, ratio)
        resampled_elevation_matrix = resample_matrix(elevation_matrix, ratio)
        resampled_azimuth_matrix = resample_matrix(azimuth_matrix, ratio)
        
        with open(filepath, 'w') as f:
            for n in range(resampled_event_matrix.shape[0]):
                for k in range(resampled_event_matrix.shape[1]):
                    if resampled_event_matrix[n, k] > 0.5:
                        elevation = int(resampled_elevation_matrix[n, k])
                        azimuth = int(resampled_azimuth_matrix[n, k])
                        f.write('{},{},{},{}\n'.format(n, k, azimuth, elevation))
            
    logging.info('    Total {} files written to {}'.format(len(list_dict), submissions_dir))
        
        
def resample_matrix(matrix, ratio):
    
    new_len = int(round(ratio * matrix.shape[0]))
    new_matrix = np.zeros((new_len, matrix.shape[1]))
    
    for n in range(new_len):
        new_matrix[n] = matrix[int(round(n / ratio))]
    
    return new_matrix
    
    
def calculate_metrics(metadata_dir, prediction_paths):
    '''Calculate metrics using official tool. This part of code is modified from:
    https://github.com/sharathadavanne/seld-dcase2019/blob/master/calculate_SELD_metrics.py
    
    Args:
      metadata_dir: string, directory of reference files. 
      prediction_paths: list of string
      
    Returns:
      metrics: dict
    '''
    
    # Load feature class
    feat_cls = cls_feature_class.FeatureClass()
    
    # Load evaluation metric class
    eval = evaluation_metrics.SELDMetrics(
        nb_frames_1s=feat_cls.nb_frames_1s(), data_gen=feat_cls)
    
    eval.reset()    # Reset the evaluation metric parameters
    for prediction_path in prediction_paths:        
        reference_path = os.path.join(metadata_dir, '{}.csv'.format(
            get_filename(prediction_path)))
        
        prediction_dict = evaluation_metrics.load_output_format_file(prediction_path)
        reference_dict = feat_cls.read_desc_file(reference_path)
    
        # Generate classification labels for SELD
        reference_tensor = feat_cls.get_clas_labels_for_file(reference_dict)
        prediction_tensor = evaluation_metrics.output_format_dict_to_classification_labels(
            prediction_dict, feat_cls)
    
        # Calculated SED and DOA scores
        eval.update_sed_scores(prediction_tensor.max(2), reference_tensor.max(2))
        eval.update_doa_scores(prediction_tensor, reference_tensor)
        
    # Overall SED and DOA scores
    sed_error_rate, sed_f1_score = eval.compute_sed_scores()
    doa_error, doa_frame_recall = eval.compute_doa_scores()
    seld_score = evaluation_metrics.compute_seld_metric(
        [sed_error_rate, sed_f1_score], [doa_error, doa_frame_recall])
    
    metrics = {
        'sed_error_rate': sed_error_rate, 
        'sed_f1_score': sed_f1_score, 
        'doa_error': doa_error, 
        'doa_frame_recall': doa_frame_recall, 
        'seld_score': seld_score }
        
    return metrics
