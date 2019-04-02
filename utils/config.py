sample_rate = 32000
window_size = 1024
hop_size = 500      # So that there are 64 frames per second
mel_bins = 64
fmin = 50       # Hz
fmax = 14000    # Hz

frames_per_second = sample_rate // hop_size
time_steps = frames_per_second * 10     # 10-second log mel spectrogram as input
submission_frames_per_second = 50   # DCASE2019 Task3 submission format

# The label configuration is the same as https://github.com/sharathadavanne/seld-dcase2019
labels = ['knock', 'drawer', 'clearthroat', 'phone', 'keysDrop', 'speech', 
    'keyboard', 'pageturn', 'cough', 'doorslam', 'laughter']
    
classes_num = len(labels)
lb_to_idx = {lb: idx for idx, lb in enumerate(labels)}
idx_to_lb = {idx: lb for idx, lb in enumerate(labels)}