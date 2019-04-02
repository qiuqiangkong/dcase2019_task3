import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))

import numpy as np
import time
import logging
import datetime
import _pickle as cPickle
import matplotlib.pyplot as plt

from utilities import (get_filename, write_submission, calculate_metrics, 
    inverse_scale)
from pytorch_utils import forward
from losses import event_spatial_loss
import config


class Evaluator(object):
    def __init__(self, model, data_generator, cuda=True):
        '''Evaluator to evaluate prediction performance. 
        
        Args: 
          model: object
          data_generator: object
          cuda: bool
        '''
        
        self.model = model
        self.data_generator = data_generator
        self.cuda = cuda
        
        self.frames_per_second = config.frames_per_second
        self.submission_frames_per_second = config.submission_frames_per_second
        
    def evaluate(self, data_type, metadata_dir, submissions_dir, 
        max_validate_num=None):
        '''Evaluate the performance. 
        
        Args: 
          data_type: 'train' | 'validate'
          metadata_dir: string, directory of reference meta csvs
          submissions_dir: string: directory to write out submission csvs
          max_validate_num: None | int, maximum iteration to run to speed up 
              evaluation
        '''
        
        # Forward
        generate_func=self.data_generator.generate_validate(
            data_type=data_type, max_validate_num=max_validate_num)
        
        list_dict = forward(
            model=self.model, 
            generate_func=generate_func, 
            cuda=self.cuda, 
            return_target=True)
        
        # Calculate loss         
        (total_loss, event_loss, position_loss) = self.calculate_loss(list_dict)
        
        logging.info('{:<20} {}: {:.3f}, {}: {:.3f}, {}: {:.3f}'
            ''.format(data_type + ' statistics: ', 'total_loss', total_loss, 
            'event_loss', event_loss, 'position_loss', position_loss))
        
        # Write out submission and evaluate using code provided by organizer
        write_submission(list_dict, submissions_dir)
        
        prediction_paths = [os.path.join(submissions_dir, 
            '{}.csv'.format(dict['name'])) for dict in list_dict]
        
        statistics = calculate_metrics(metadata_dir, prediction_paths)
        
        for key in statistics.keys():
            logging.info('    {:<20} {:.3f}'.format(key + ' :', statistics[key]))
            
        return statistics
                    
    def calculate_loss(self, list_dict):
        total_loss_list = []
        event_loss_list = []
        position_loss_list = []
        
        for dict in list_dict:
            (output_dict, target_dict) = self._get_output_target_dict(dict)
            
            (total_loss, event_loss, position_loss) = event_spatial_loss(
                output_dict=output_dict, 
                target_dict=target_dict, 
                return_individual_loss=True)
            
            total_loss_list.append(total_loss)
            event_loss_list.append(event_loss)
            position_loss_list.append(position_loss)

        return np.mean(total_loss_list), np.mean(event_loss_list), np.mean(position_loss_list)
        
    def _get_output_target_dict(self, dict):
        output_dict = {
            'event': dict['output_event'], 
            'elevation': dict['output_elevation'], 
            'azimuth': dict['output_azimuth']}
            
        target_dict = {
            'event': dict['target_event'], 
            'elevation': dict['target_elevation'], 
            'azimuth': dict['target_azimuth']}
            
        return output_dict, target_dict
        
            
    def visualize(self, data_type, max_validate_num=None):
        '''Visualize the log mel spectrogram, reference and prediction of 
        sound events, elevation and azimuth. 
        
        Args:
          data_type: 'train' | 'validate'
          max_validate_num: None | int, maximum iteration to run to speed up evaluation
        '''
        
        mel_bins = config.mel_bins
        frames_per_second = config.frames_per_second
        classes_num = config.classes_num
        labels = config.labels
        
        # Forward
        generate_func=self.data_generator.generate_validate(
            data_type=data_type, max_validate_num=max_validate_num)
            
        list_dict = forward(
            model=self.model, 
            generate_func=generate_func, 
            cuda=self.cuda, 
            return_input=True, 
            return_target=True)

        for n, dict in enumerate(list_dict):
            
            print('File: {}'.format(dict['name']))

            frames_num = dict['target_event'].shape[1]
            length_in_second = frames_num / float(frames_per_second)
            
            fig, axs = plt.subplots(4, 2, figsize=(15, 10))
            logmel = inverse_scale(dict['feature'][0][0], 
                self.data_generator.scalar['mean'], 
                self.data_generator.scalar['std'])
            axs[0, 0].matshow(logmel.T, origin='lower', aspect='auto', cmap='jet')
            axs[1, 0].matshow(dict['target_event'][0].T, origin='lower', aspect='auto', cmap='jet')
            axs[2, 0].matshow(dict['output_event'][0].T, origin='lower', aspect='auto', cmap='jet')
            axs[0, 1].matshow(dict['target_elevation'][0].T, origin='lower', aspect='auto', cmap='jet')
            axs[1, 1].matshow(dict['target_azimuth'][0].T, origin='lower', aspect='auto', cmap='jet')
            masksed_evaluation = dict['output_elevation'] * dict['output_event']
            axs[2, 1].matshow(masksed_evaluation[0].T, origin='lower', aspect='auto', cmap='jet')
            masksed_azimuth = dict['output_azimuth'] * dict['output_event']
            axs[3, 1].matshow(masksed_azimuth[0].T, origin='lower', aspect='auto', cmap='jet')
            
            axs[0,0].set_title('Log mel spectrogram', color='r')
            axs[1,0].set_title('Reference sound events', color='r')
            axs[2,0].set_title('Predicted sound events', color='b')
            axs[0,1].set_title('Reference elevation', color='r')
            axs[1,1].set_title('Reference azimuth', color='r')
            axs[2,1].set_title('Predicted elevation', color='b')
            axs[3,1].set_title('Predicted azimuth', color='b')
            
            for i in range(4):
                for j in range(2):
                    axs[i, j].set_xticks([0, frames_num])
                    axs[i, j].set_xticklabels(['0', '{:.1f} s'.format(length_in_second)])
                    axs[i, j].xaxis.set_ticks_position('bottom')
                    axs[i, j].set_yticks(np.arange(classes_num))
                    axs[i, j].set_yticklabels(labels)
                    axs[i, j].yaxis.grid(color='w', linestyle='solid', linewidth=0.2)
            
            axs[0, 0].set_ylabel('Mel bins')
            axs[0, 0].set_yticks([0, mel_bins])
            axs[0, 0].set_yticklabels([0, mel_bins])
            
            axs[3, 0].set_visible(False)
            
            fig.tight_layout()
            plt.show()
            
            
class StatisticsContainer(object):
    def __init__(self, statistics_path):
        '''Container of statistics during training. 
        
        Args:
          statistics_path: string, path to write out
        '''
        self.statistics_path = statistics_path

        self.backup_statistics_path = '{}_{}.pickle'.format(
            os.path.splitext(self.statistics_path)[0], 
                datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

        self.statistics_list = []

    def append_and_dump(self, iteration, statistics):
        '''Append statistics to container and dump the container. 
        
        Args:
          iteration: int
          statistics: dict of statistics
        '''
        statistics['iteration'] = iteration
        self.statistics_list.append(statistics)

        cPickle.dump(self.statistics_list, open(self.statistics_path, 'wb'))
        cPickle.dump(self.statistics_list, open(self.backup_statistics_path, 'wb'))
        logging.info('    Dump statistics to {}'.format(self.statistics_path))