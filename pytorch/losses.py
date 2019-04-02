import torch
import torch.nn.functional as F


def to_tensor(x):
    if type(x).__name__ == 'ndarray':
        return torch.Tensor(x)
    else:
        return x


def binary_crossentropy(output, target):
    '''Binary crossentropy between output and target. 
    
    Args:
      output: (batch_size, frames_num, classes_num)
      target: (batch_size, frames_num, classes_num)
    '''
    output = to_tensor(output)
    target = to_tensor(target)
    
    # To let output and target to have the same time steps. The mismatching 
    # size is caused by pooling in CNNs. 
    N = min(output.shape[1], target.shape[1])
    
    return F.binary_cross_entropy(
        output[:, 0 : N, :], 
        target[:, 0 : N, :])


def mean_absolute_error(output, target, mask):
    '''Mean absolute error between output and target. 
    
    Args:
      output: (batch_size, frames_num, classes_num)
      target: (batch_size, frames_num, classes_num)
    '''    
    output = to_tensor(output)
    target = to_tensor(target)
    mask = to_tensor(mask)
    
    # To let output and target to have the same time steps. The mismatching 
    # size is caused by pooling in CNNs. 
    N = min(output.shape[1], target.shape[1])
    
    output = output[:, 0 : N, :]
    target = target[:, 0 : N, :]
    mask = mask[:, 0 : N, :]
    
    normalize_value = torch.sum(mask)
    
    return torch.sum(torch.abs(output - target) * mask) / normalize_value


def event_spatial_loss(output_dict, target_dict, return_individual_loss=False):
    '''Joint event and spatial loss. 
    
    Args:
      output_dict: {'event': (batch_size, frames_num, classes_num), 
                    'elevation': (batch_size, frames_num, classes_num), 
                    'azimuth': (batch_size, frames_num, classes_num)}
      target_dict: {'event': (batch_size, frames_num, classes_num), 
                    'elevation': (batch_size, frames_num, classes_num), 
                    'azimuth': (batch_size, frames_num, classes_num)}
      return_individual_loss: bool
      
    Returns:
      total_loss: scalar
    '''
    
    event_loss = binary_crossentropy(
        output_dict['event'], 
        target_dict['event'])
        
    elevation_loss = mean_absolute_error(
        output=output_dict['elevation'], 
        target=target_dict['elevation'], 
        mask=target_dict['event'])
        
    azimuth_loss = mean_absolute_error(
        output=output_dict['azimuth'], 
        target=target_dict['azimuth'], 
        mask=target_dict['event'])
    
    alpha = 0.01    # To control the balance between the event loss and position loss
    position_loss = alpha * (elevation_loss + azimuth_loss)
    
    total_loss = event_loss + position_loss
    
    if return_individual_loss:
        return total_loss, event_loss, position_loss
    else:
        return total_loss