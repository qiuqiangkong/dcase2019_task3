import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], 'utils'))
import numpy as np
import pandas as pd
import argparse
import h5py
import librosa
from scipy import signal
import matplotlib.pyplot as plt
import time
import csv
import random

from utilities import (read_multichannel_audio, create_folder, 
    calculate_scalar_of_tensor)
import config


class LogMelExtractor(object):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, fmax):
        '''Log mel feature extractor. 
        
        Args:
          sample_rate: int
          window_size: int
          hop_size: int
          mel_bins: int
          fmin: int, minimum frequency of mel filter banks
          fmax: int, maximum frequency of mel filter banks
        '''
        
        self.window_size = window_size
        self.hop_size = hop_size
        self.window_func = np.hanning(window_size)
        
        self.melW = librosa.filters.mel(
            sr=sample_rate, 
            n_fft=window_size, 
            n_mels=mel_bins, 
            fmin=fmin, 
            fmax=fmax).T
        '''(n_fft // 2 + 1, mel_bins)'''
    
    def transform_multichannel(self, multichannel_audio):
        '''Extract feature of a multichannel audio file. 
        
        Args:
          multichannel_audio: (samples, channels_num)
          
        Returns:
          feature: (channels_num, frames_num, freq_bins)
        '''
        
        (samples, channels_num) = multichannel_audio.shape
        
        feature = np.array([self.transform_singlechannel(
            multichannel_audio[:, m]) for m in range(channels_num)])
        
        return feature
    
    def transform_singlechannel(self, audio):
        '''Extract feature of a singlechannel audio file. 
        
        Args:
          audio: (samples,)
          
        Returns:
          feature: (frames_num, freq_bins)
        '''
    
        window_size = self.window_size
        hop_size = self.hop_size
        window_func = self.window_func
        
        # Compute short-time Fourier transform
        stft_matrix = librosa.core.stft(
            y=audio, 
            n_fft=window_size, 
            hop_length=hop_size, 
            window=window_func, 
            center=True, 
            dtype=np.complex64, 
            pad_mode='reflect').T
        '''(N, n_fft // 2 + 1)'''
    
        # Mel spectrogram
        mel_spectrogram = np.dot(np.abs(stft_matrix) ** 2, self.melW)
        
        # Log mel spectrogram
        logmel_spectrogram = librosa.core.power_to_db(
            mel_spectrogram, ref=1.0, amin=1e-10, 
            top_db=None)
        
        logmel_spectrogram = logmel_spectrogram.astype(np.float32)
        
        return logmel_spectrogram

 
def calculate_feature_for_each_audio_file(args):
    '''Calculate feature for each audio file and write out to hdf5. 
    
    Args:
      dataset_dir: string
      workspace: string
      data_type: 'development' | 'evaluation'
      audio_type: 'foa' | 'mic'
      mini_data: bool, set True for debugging on a small part of data
    '''
    
    # Arguments & parameters
    dataset_dir = args.dataset_dir
    workspace = args.workspace
    data_type = args.data_type
    audio_type = args.audio_type
    mini_data = args.mini_data
    
    sample_rate = config.sample_rate
    window_size = config.window_size
    hop_size = config.hop_size
    mel_bins = config.mel_bins
    fmin = config.fmin
    fmax = config.fmax
    frames_per_second = config.frames_per_second
    
    # Paths
    if data_type == 'development':
        data_type = 'dev'
        
    elif data_type == 'evaluation':
        data_type = 'eva'
        raise Exception('Todo after evaluation data released. ')
        
    if mini_data:
        prefix = 'minidata_'
    else:
        prefix = ''
    
    metas_dir = os.path.join(dataset_dir, 'metadata_{}'.format(data_type))
    audios_dir = os.path.join(dataset_dir, '{}_{}'.format(audio_type, data_type))
    
    features_dir = os.path.join(workspace, 'features', 
        '{}{}_{}_logmel_{}frames_{}melbins'.format(prefix, audio_type, 
        data_type, frames_per_second, mel_bins))
        
    create_folder(features_dir)
    
    # Feature extractor
    feature_extractor = LogMelExtractor(
        sample_rate=sample_rate, 
        window_size=window_size, 
        hop_size=hop_size, 
        mel_bins=mel_bins, 
        fmin=fmin, 
        fmax=fmax)
    
    # Extract features and targets
    meta_names = sorted(os.listdir(metas_dir))
    
    if mini_data:
        random_state = np.random.RandomState(1234)
        random_state.shuffle(meta_names)
    
    print('Extracting features of all audio files ...')
    extract_time = time.time()
    
    for (n, meta_name) in enumerate(meta_names):
        
        meta_path = os.path.join(metas_dir, meta_name)
        bare_name = os.path.splitext(meta_name)[0]
        audio_path = os.path.join(audios_dir, '{}.wav'.format(bare_name))
        feature_path = os.path.join(features_dir, '{}.h5'.format(bare_name))
        
        df = pd.read_csv(meta_path, sep=',')
        event = df['sound_event_recording'].values
        start_time = df['start_time'].values
        end_time = df['end_time'].values
        elevation = df['ele'].values
        azimuth = df['azi'].values
        distance = df['dist'].values
        
        # Read audio
        (multichannel_audio, _) = read_multichannel_audio(
            audio_path=audio_path, 
            target_fs=sample_rate)
        
        # Extract feature
        feature = feature_extractor.transform_multichannel(multichannel_audio)
        
        with h5py.File(feature_path, 'w') as hf:
            
            hf.create_dataset('feature', data=feature, dtype=np.float32)
            
            hf.create_group('target')
            hf['target'].create_dataset('event', data=[e.encode() for e in event], dtype='S20')
            hf['target'].create_dataset('start_time', data=start_time, dtype=np.float32)
            hf['target'].create_dataset('end_time', data=end_time, dtype=np.float32)
            hf['target'].create_dataset('elevation', data=elevation, dtype=np.int32)
            hf['target'].create_dataset('azimuth', data=azimuth, dtype=np.int32)
            hf['target'].create_dataset('distance', data=distance, dtype=np.int32)
        
        print(n, feature_path, feature.shape)
    
        if mini_data and n == 10:
            break
    
    print('Extract features finished! {:.3f} s'.format(time.time() - extract_time))
    
    
def calculate_scalar(args):
    '''Calculate and write out scalar of development data. 
    
    Args:
      dataset_dir: string
      workspace: string
      audio_type: 'foa' | 'mic'
      mini_data: bool, set True for debugging on a small part of data
    '''
    
    # Arguments & parameters
    workspace = args.workspace
    audio_type = args.audio_type
    mini_data = args.mini_data
    data_type = 'dev'
    
    mel_bins = config.mel_bins
    frames_per_second = config.frames_per_second
    
    # Paths
    if mini_data:
        prefix = 'minidata_'
    else:
        prefix = ''
    
    features_dir = os.path.join(workspace, 'features', 
        '{}{}_{}_logmel_{}frames_{}melbins'.format(prefix, audio_type, 
        data_type, frames_per_second, mel_bins))
        
    scalar_path = os.path.join(workspace, 'scalars', 
        '{}{}_{}_logmel_{}frames_{}melbins'.format(prefix, audio_type, 
        data_type, frames_per_second, mel_bins), 'scalar.h5')
        
    create_folder(os.path.dirname(scalar_path))
        
    # Load data
    load_time = time.time()
    feature_names = os.listdir(features_dir)
    all_features = []
    
    for feature_name in feature_names:
        feature_path = os.path.join(features_dir, feature_name)
        
        with h5py.File(feature_path, 'r') as hf:
            feature = hf['feature'][:]
            all_features.append(feature)
            
    print('Load feature time: {:.3f} s'.format(time.time() - load_time))
    
    # Calculate scalar
    all_features = np.concatenate(all_features, axis=1)
    (mean, std) = calculate_scalar_of_tensor(all_features)
    
    with h5py.File(scalar_path, 'w') as hf:
        hf.create_dataset('mean', data=mean, dtype=np.float32)
        hf.create_dataset('std', data=std, dtype=np.float32)
    
    print('All features: {}'.format(all_features.shape))
    print('mean: {}'.format(mean))
    print('std: {}'.format(std))
    print('Write out scalar to {}'.format(scalar_path))
            

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='')
    subparsers = parser.add_subparsers(dest='mode')

    parser_logmel = subparsers.add_parser('calculate_feature_for_each_audio_file')
    parser_logmel.add_argument('--dataset_dir', type=str, required=True)
    parser_logmel.add_argument('--workspace', type=str, required=True)
    parser_logmel.add_argument('--data_type', type=str, required=True, choices=['development', 'evaluation'])
    parser_logmel.add_argument('--audio_type', type=str, required=True, choices=['foa', 'mic'])
    parser_logmel.add_argument('--mini_data', action='store_true', default=False)

    parser_scalar = subparsers.add_parser('calculate_scalar')
    parser_scalar.add_argument('--workspace', type=str, required=True)
    parser_scalar.add_argument('--data_type', type=str, required=True, choices=['development', 'evaluation'])
    parser_scalar.add_argument('--audio_type', type=str, required=True, choices=['foa', 'mic'])
    parser_scalar.add_argument('--mini_data', action='store_true', default=False)
    
    args = parser.parse_args()
    
    if args.mode == 'calculate_feature_for_each_audio_file':
        calculate_feature_for_each_audio_file(args)
        
    elif args.mode == 'calculate_scalar':
        calculate_scalar(args)
        
    else:
        raise Exception('Incorrect arguments!')