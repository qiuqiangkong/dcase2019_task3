#!/bin/bash
# You need to modify this path to your downloaded dataset directory
DATASET_DIR='/vol/vssp/cvpnobackup/scratch_4weeks/qk00006/dcase2019/task3/dataset_root'

# You need to modify this path to your workspace to store features, models, etc.
WORKSPACE='/vol/vssp/msos/qk/workspaces/dcase2019_task3'

# Hyper-parameters
GPU_ID=1
DATA_TYPE='development' # 'development' | 'evaluation'
AUDIO_TYPE='foa'        # 'foa' | 'mic'
MODEL_TYPE='Cnn_9layers'
BATCH_SIZE=32

# Calculate feature
python utils/features.py calculate_feature_for_each_audio_file --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --data_type=$DATA_TYPE --audio_type=$AUDIO_TYPE

# Calculate scalar
python utils/features.py calculate_scalar --workspace=$WORKSPACE --data_type=$DATA_TYPE --audio_type=$AUDIO_TYPE

############ Train and validate system on development dataset ############
for HOLDOUT_FOLD in '1' '2' '3' '4'
  do
  echo 'Holdout fold: '$HOLDOUT_FOLD

  # Train
  CUDA_VISIBLE_DEVICES=$GPU_ID python pytorch/main.py train --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --audio_type=$AUDIO_TYPE --holdout_fold=$HOLDOUT_FOLD --model_type=$MODEL_TYPE --batch_size=$BATCH_SIZE --cuda

  # Validate
  CUDA_VISIBLE_DEVICES=$GPU_ID python pytorch/main.py inference_validation --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --audio_type=$AUDIO_TYPE --holdout_fold=$HOLDOUT_FOLD --model_type=$MODEL_TYPE --iteration=5000 --batch_size=$BATCH_SIZE --cuda

  HOLDOUT_FOLD=$[$HOLDOUT_FOLD+1]
done

# Calculate metrics on all cross-validation folds
HOLDOUT_FOLD=-1
CUDA_VISIBLE_DEVICES=$GPU_ID python pytorch/main.py inference_validation --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --audio_type=$AUDIO_TYPE --holdout_fold=$HOLDOUT_FOLD --model_type=$MODEL_TYPE --iteration=5000 --batch_size=$BATCH_SIZE --cuda

############ END ############
