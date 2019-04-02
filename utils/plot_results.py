import argparse
import os
import matplotlib.pyplot as plt
import _pickle as cPickle
import numpy as np

import config


def plot_results(args):
    # Arugments & parameters
    dataset_dir = args.dataset_dir
    workspace = args.workspace
    audio_type = args.audio_type
    
    filename = 'main'
    prefix = ''
    frames_per_second = config.frames_per_second
    mel_bins = config.mel_bins
    holdout_fold = 1
    max_plot_iteration = 5000
    
    
    iterations = np.arange(0, max_plot_iteration, 200)
    
    def _load_stat(model_type):
        validate_statistics_path = os.path.join(workspace, 'statistics', filename, 
            '{}{}_{}_logmel_{}frames_{}melbins'.format(prefix, audio_type, 
            'dev', frames_per_second, mel_bins), model_type, 'validate_statistics.pickle')
        
        statistics_list = cPickle.load(open(validate_statistics_path, 'rb'))
        
        sed_error_rate = np.array([statistics['sed_error_rate'] for statistics in statistics_list])
        sed_f1_score = np.array([statistics['sed_f1_score'] for statistics in statistics_list])
        doa_error = np.array([statistics['doa_error'] for statistics in statistics_list])
        doa_frame_recall = np.array([statistics['doa_frame_recall'] for statistics in statistics_list])
        seld_score = np.array([statistics['seld_score'] for statistics in statistics_list])
        
        legend = '{}'.format(model_type)
        
        results = {'sed_error_rate': sed_error_rate, 
            'sed_f1_score': sed_f1_score, 'doa_error': doa_error, 
            'doa_frame_recall': doa_frame_recall, 'seld_score': seld_score, 
            'legend': legend}
        
        print('Model type: {}'.format(model_type))
        print('    sed_error_rate: {:.3f}'.format(sed_error_rate[-1]))
        print('    sed_f1_score: {:.3f}'.format(sed_f1_score[-1]))
        print('    doa_error: {:.3f}'.format(doa_error[-1]))
        print('    doa_frame_recall: {:.3f}'.format(doa_frame_recall[-1]))
        print('    seld_score: {:.3f}'.format(seld_score[-1]))
        
        return results
    
    measure_keys = ['sed_error_rate', 'sed_f1_score', 'doa_error', 'doa_frame_recall']
    
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        
    results_dict = {}
    results_dict['Cnn_5layers_AvgPooling'] = _load_stat('Cnn_5layers_AvgPooling')
    results_dict['Cnn_9layers_AvgPooling'] = _load_stat('Cnn_9layers_AvgPooling')
    results_dict['Cnn_9layers_MaxPooling'] = _load_stat('Cnn_9layers_MaxPooling')
    results_dict['Cnn_13layers_AvgPooling'] = _load_stat('Cnn_13layers_AvgPooling')
        
    for n, measure_key in enumerate(measure_keys):
        lines = []
        
        for model_key in results_dict.keys():
            line, = axs[n // 2, n % 2].plot(results_dict[model_key][measure_key], label=results_dict[model_key]['legend'])
            lines.append(line)
            
        axs[n // 2, n % 2].set_title(measure_key)
        axs[n // 2, n % 2].legend(handles=lines, loc=4)
        axs[n // 2, n % 2].set_ylim(0, 1.0)
        axs[n // 2, n % 2].set_xlabel('Iterations')
        axs[n // 2, n % 2].grid(color='b', linestyle='solid', linewidth=0.2)
        axs[n // 2, n % 2].xaxis.set_ticks(np.arange(0, len(iterations), len(iterations) // 4))
        axs[n // 2, n % 2].xaxis.set_ticklabels(np.arange(0, max_plot_iteration, max_plot_iteration // 4))

    axs[1, 0].set_ylim(0, 100.)
    axs[0, 0].set_ylabel('sed_error_rate')
    axs[0, 1].set_ylabel('sed_f1_score')
    axs[1, 0].set_ylabel('doa_error')
    axs[1, 1].set_ylabel('doa_frame_recall')
        
    plt.tight_layout()
    fig_path = '_tmp.png'
    plt.savefig(fig_path)
    print('Save fig to {}'.format(fig_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--workspace', type=str, required=True)
    parser.add_argument('--audio_type', type=str, choices=['foa', 'mic'], required=True)

    args = parser.parse_args()
    
    plot_results(args)