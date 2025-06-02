 # for saving/loading data in HDF5 format (used here for model weights)

import h5py
import torch
import shutil

## save model weights -> save weights 
def save_net(fname, net): # Save model weights into an HDF5 (.h5) file
    with h5py.File(fname, 'w') as h5f: # Loop through all layers in the model and save their weights
        for k, v in net.state_dict().items():
            h5f.create_dataset(k, data=v.cpu().numpy())

## load model weights -> copy saved weights into model
def load_net(fname, net): # Load model weights from an HDF5 (.h5) file into the model
    with h5py.File(fname, 'r') as h5f: # Loop through all layers and copy saved weights back into the model
        for k, v in net.state_dict().items():        
            param = torch.from_numpy(np.asarray(h5f[k]))         
            v.copy_(param)
            
## save training checkpoint -> save current state -> save the best model 
def save_checkpoint(state, is_best,task_id, filename='checkpoint.pth.tar'): # Save a training checkpoint (PyTorch format)
    torch.save(state, task_id+filename) # Save current state to a file (for resuming training later)
    if is_best:
        shutil.copyfile(task_id+filename, task_id+'model_best.pth.tar')  # If this is the best model so far, also save a copy named 'model_best.pth.tar'           

