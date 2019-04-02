import numpy as np
import h5py
import csv
import time
import logging
import os
import glob
import matplotlib.pyplot as plt
import logging

from utilities import scale
import config


class DataGenerator(object):

    def __init__(self, features_dir, scalar, batch_size, holdout_fold, seed=1234):
        '''Data generator for training and validation. 
        
        Args:
          features_dir: string, directory of features
          scalar: object, containing mean and std value
          batch_size: int
          holdout_fold: '1' | '2' | '3' | '4' | 'none', where 'none' indicates 
              using all data without validation for training
          seed: int, random seed
        '''

        self.scalar = scalar
        self.batch_size = batch_size
        self.random_state = np.random.RandomState(seed)

        self.frames_per_second = config.frames_per_second
        self.classes_num = config.classes_num
        self.lb_to_idx = config.lb_to_idx
        self.time_steps = config.time_steps
        
        # Load data
        load_time = time.time()
        
        feature_names = sorted(os.listdir(features_dir))
        
        self.train_feature_names = [name for name in feature_names \
            if 'split{}'.format(holdout_fold) not in name]
            
        self.validate_feature_names = [name for name in feature_names \
            if 'split{}'.format(holdout_fold) in name]

        self.train_features_list = []
        self.train_event_matrix_list = []
        self.train_elevation_matrix_list = []
        self.train_azimuth_matrix_list = []
        self.train_index_array_list = []
        frame_index = 0
        
        # Load training feature and targets
        for feature_name in self.train_feature_names:
            feature_path = os.path.join(features_dir, feature_name)
            
            (feature, event_matrix, elevation_matrix, azimuth_matrix) = \
                self.load_hdf5(feature_path)
                
            frames_num = feature.shape[1]   
            '''Number of frames of the log mel spectrogram of an audio 
            recording. May be different from file to file'''
            
            index_array = np.arange(frame_index, frame_index + frames_num - self.time_steps)
            frame_index += frames_num
            
            # Append data
            self.train_features_list.append(feature)
            self.train_event_matrix_list.append(event_matrix)
            self.train_elevation_matrix_list.append(elevation_matrix)
            self.train_azimuth_matrix_list.append(azimuth_matrix)
            self.train_index_array_list.append(index_array)

        self.train_features = np.concatenate(self.train_features_list, axis=1)
        self.train_event_matrix = np.concatenate(self.train_event_matrix_list, axis=0)
        self.train_elevation_matrix = np.concatenate(self.train_elevation_matrix_list, axis=0)
        self.train_azimuth_matrix = np.concatenate(self.train_azimuth_matrix_list, axis=0)
        self.train_index_array = np.concatenate(self.train_index_array_list, axis=0)
            
        # Load validation feature and targets
        self.validate_features_list = []
        self.validate_event_matrix_list = []
        self.validate_elevation_matrix_list = []
        self.validate_azimuth_matrix_list = []
        
        for feature_name in self.validate_feature_names:
            feature_path = os.path.join(features_dir, feature_name)
            
            (feature, event_matrix, elevation_matrix, azimuth_matrix) = \
                self.load_hdf5(feature_path)
                
            self.validate_features_list.append(feature)
            self.validate_event_matrix_list.append(event_matrix)
            self.validate_elevation_matrix_list.append(elevation_matrix)
            self.validate_azimuth_matrix_list.append(azimuth_matrix)
            
        logging.info('Load data time: {:.3f} s'.format(time.time() - load_time))
        logging.info('Training audio num: {}'.format(len(self.train_feature_names)))
        logging.info('Validation audio num: {}'.format(len(self.validate_feature_names)))
        
        self.random_state.shuffle(self.train_index_array)
        self.pointer = 0
        
    def load_hdf5(self, feature_path):
        '''Load hdf5. 
        
        Args:
          feature_path: string
          
        Returns:
          feature: (channels_num, frames_num, freq_bins)
          eevnt_matrix: (frames_num, classes_num)
          elevation_matrix: (frames_num, classes_num)
          azimuth_matrix: (frames_num, classes_num)
        '''
        
        with h5py.File(feature_path, 'r') as hf:
            feature = hf['feature'][:]
            events = [e.decode() for e in hf['target']['event'][:]]
            start_times = hf['target']['start_time'][:]
            end_times = hf['target']['end_time'][:]
            elevations = hf['target']['elevation'][:]
            azimuths = hf['target']['azimuth'][:]
            distances = hf['target']['distance'][:]
        
        frames_num = feature.shape[1]
        
        # Researve space data
        event_matrix = np.zeros((frames_num, self.classes_num))
        elevation_matrix = np.zeros((frames_num, self.classes_num))
        azimuth_matrix = np.zeros((frames_num, self.classes_num))
        
        for n in range(len(events)):
            class_id = self.lb_to_idx[events[n]]
            start_frame = int(round(start_times[n] * self.frames_per_second))
            end_frame = int(round(end_times[n] * self.frames_per_second)) + 1
            
            event_matrix[start_frame : end_frame, class_id] = 1
            elevation_matrix[start_frame : end_frame, class_id] = elevations[n]
            azimuth_matrix[start_frame : end_frame, class_id] = azimuths[n]
         
        return feature, event_matrix, elevation_matrix, azimuth_matrix
    
    def generate_train(self):
        '''Generate mini-batch data for training. 
        
        Returns:
          batch_data_dict: dict containing feature, event, elevation and azimuth
        '''

        while True:
            # Reset pointer
            if self.pointer >= len(self.train_index_array):
                self.pointer = 0
                self.random_state.shuffle(self.train_index_array)

            # Get batch indexes
            batch_indexes = self.train_index_array[
                self.pointer: self.pointer + self.batch_size]
                
            data_indexes = batch_indexes[:, None] + np.arange(self.time_steps)
            
            self.pointer += self.batch_size

            batch_feature = self.train_features[:, data_indexes]
            batch_event_matrix = self.train_event_matrix[data_indexes]
            batch_elevation_matrix = self.train_elevation_matrix[data_indexes]
            batch_azimuth_matrix = self.train_azimuth_matrix[data_indexes]

            # Transform data
            batch_feature = self.transform(batch_feature)

            batch_data_dict = {
                'feature': batch_feature, 
                'event': batch_event_matrix, 
                'elevation': batch_elevation_matrix, 
                'azimuth': batch_azimuth_matrix}

            yield batch_data_dict
            
    def generate_validate(self, data_type, max_validate_num=None):
        '''Generate feature and targets of a single audio file. 
        
        Args:
          data_type: 'train' | 'validate'
          max_validate_num: None | int, maximum iteration to run to speed up 
              evaluation
        
        Returns:
          batch_data_dict: dict containing feature, event, elevation and azimuth
        '''
        
        if data_type == 'train':
            feature_names = self.train_feature_names
            features_list = self.train_features_list
            event_matrix_list = self.train_event_matrix_list
            elevation_matrix_list = self.train_elevation_matrix_list
            azimuth_matrix_list = self.train_azimuth_matrix_list
            
        elif data_type == 'validate':
            feature_names = self.validate_feature_names
            features_list = self.validate_features_list
            event_matrix_list = self.validate_event_matrix_list
            elevation_matrix_list = self.validate_elevation_matrix_list
            azimuth_matrix_list = self.validate_azimuth_matrix_list
        
        else:
            raise Exception('Incorrect argument!')
        
        validate_num = len(feature_names)
        
        for n in range(validate_num):
            if n == max_validate_num:
                break
            
            name = os.path.splitext(feature_names[n])[0]
            feature = features_list[n]
            event_matrix = event_matrix_list[n]
            elevation_matrix = elevation_matrix_list[n]
            azimuth_matrix = azimuth_matrix_list[n]
            
            feature = self.transform(feature)

            batch_data_dict = {
                'name': name, 
                'feature': feature[:, None, :, :], # (channels_num, batch_size=1, frames_num, mel_bins)
                'event': event_matrix[None, :, :], # (batch_size=1, frames_num, mel_bins)
                'elevation': elevation_matrix[None, :, :], # (batch_size=1, frames_num, mel_bins)
                'azimuth': azimuth_matrix[None, :, :]   # (batch_size=1, frames_num, mel_bins)
                }
            '''The None above indicates using an entire audio recording as 
            input and batch_size=1 in inference'''
                
            yield batch_data_dict
            
    def transform(self, x):
        return scale(x, self.scalar['mean'], self.scalar['std'])
    