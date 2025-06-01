# change the csrnet_cbam to csrnet at line 5, 24, 113 if using csrnet only
import sys
import os
import warnings
from model import CSRNet_CBAM
from utils import save_checkpoint
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
import numpy as np
import argparse
import json
import cv2
import dataset
import time
import kornia
from kornia import metrics
import torch.nn.functional as F
import matplotlib.pyplot as plt
from kornia.metrics import ssim, psnr

# parser for command line arguments
parser = argparse.ArgumentParser(description='PyTorch CSRNet_CBAM')
parser.add_argument('train_json', metavar='TRAIN', help='path to train json')
parser.add_argument('test_json', metavar='TEST', help='path to test json')
parser.add_argument('--pre', '-p', metavar='PRETRAINED', default=None, type=str,
                    help='path to the pretrained model')
parser.add_argument('gpu', metavar='GPU', type=str, help='GPU id to use.')
parser.add_argument('task', metavar='TASK', type=str, help='task id to use.')

ALPHA = 1.0  # Weight for hybrid loss
BETA = 0.1   # Weight for count loss

class SSIMLoss(nn.Module): # SSIM loss for density maps
    def __init__(self, window_size=11):
        super().__init__()
        self.window_size = window_size

    def forward(self, pred, target):
        if pred.dim() == 3:
            pred = pred.unsqueeze(0)
        if target.dim() == 3:
            target = target.unsqueeze(0)
        return 1- metrics.ssim(pred, target, window_size=self.window_size)

class HybridLoss(nn.Module): # combines MSE and SSIM
    def __init__(self, window_size=11):
        super().__init__()
        self.mse = nn.MSELoss()
        self.ssim = SSIMLoss(window_size)

    def forward(self, pred, target):
        mse_part = self.mse(pred, target)
        ssim_part = self.ssim(pred, target)
        return mse_part + ssim_part
        
# MAE between total predicted and ground truth counts
def count_loss(pred, target): # Count Loss
    pred_count = pred.sum(dim=[1, 2, 3])
    target_count = target.sum(dim=[1, 2, 3])
    #return F.mse_loss(pred_count, target_count)
    return F.l1_loss(pred_count, target_count)

def main():

    # # Tracking lists for metrics and accuracy
    mae_list = []
    rmse_list = []
    ssim_list = []
    psnr_list = []

    loss_history = []
    
    train_loss_list = []
    #val_loss_list = []
    
    train_acc_list = []
    val_acc_list = []

    best_mae = float('inf') 
    best_ssim = -1.0         
    best_psnr = -1.0
    best_model_path = 'best_model.pth'
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    global args
    
    # Training parameters
    args = parser.parse_args()
    args.original_lr = 1e-5
    args.lr = 1e-5
    args.batch_size = 1
    #args.momentum = 0.95
    args.decay = 5*1e-4
    args.start_epoch = 0
    args.epochs = 30
    args.steps = [10,20]
    args.scales = [1, 0.5, 0.1]
    args.workers = 4
    args.seed = int(time.time())
    args.print_freq = 100
    
    # Load train and test data lists
    with open(args.train_json, 'r') as outfile:        
        train_list = json.load(outfile)[:200]
    with open(args.test_json, 'r') as outfile:       
        val_list = json.load(outfile)[:50] 

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    torch.cuda.manual_seed(args.seed) if torch.cuda.is_available() else None
    
    model = CSRNet_CBAM().to(device)
    criterion = HybridLoss(window_size=11).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)

    # Load pretrained model if provided
    if args.pre and os.path.isfile(args.pre):
        print(f"=> loading checkpoint '{args.pre}'")
        checkpoint = torch.load(args.pre)
        args.start_epoch = checkpoint['epoch']
        best_mae = checkpoint['best_mae']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"=> loaded checkpoint '{args.pre}' (epoch {checkpoint['epoch']})")
    elif args.pre:
        print(f"=> no checkpoint found at '{args.pre}'")
            
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # Training step
        train_loss, train_acc = train(train_list, model, criterion, optimizer, epoch, device, loss_history)
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        
        # Evaluate step
        mae, rmse, ssim, psnr,val_acc = eval_metrics(val_list, model, device)
        mae_list.append(mae)
        rmse_list.append(rmse)
        ssim_list.append(ssim)
        psnr_list.append(psnr)
        val_acc_list.append(val_acc)

        
        # Save best model
        is_best = mae < best_mae
        
        if is_best:
            best_mae = mae
            best_ssim = ssim
            best_psnr = psnr
            print(f"Saved best model at epoch {epoch} with MAE: {mae:.3f}, SSIM: {ssim:.3f}, PSNR: {psnr:.3f}")
            
            torch.save({
                'epoch': epoch + 1,
                'arch': args.pre,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'mae': mae,
                'rmse': rmse,
                'ssim': ssim,
                'psnr': psnr,
                'train_loss':train_loss,
            }, f'pretrained_model.pth')
        
        print(f' * Best MAE so far: {best_mae:.3f}')
        
        # Save checkpoint every epoch
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.pre,
            'model_state_dict': model.state_dict(),
            'best_mae': best_mae,
            'optimizer': optimizer.state_dict(),
        }, is_best, args.task)

    # Plots and summaries
    print("mae_list=", mae_list)
    print("rmse_list=", rmse_list)
    print("ssim_list=", ssim_list)
    print("psnr_list=", psnr_list)

    print("\n ==== Best Metrics Summary ====")
    print(f"Best MAE : {min(mae_list):.3f}")
    print(f"Best RMSE : {min(rmse_list):.3f}")
    print(f"Best SSIM : {min(ssim_list):.3f}")
    print(f"Best PSNR : {min(psnr_list):.3f}")

    epochs = range(1, len(mae_list) + 1) 

    # --------- Graph: Loss History ----------
    plt.figure(figsize=(12, 4))
    plt.plot(train_loss_list)
    plt.title("Loss history CSRNet+Faster Gaussian") 
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.savefig("01. Loss_history.png") 
    print("Plot saved as 1. Loss_history.png")
    
    # --------- Graph: Count Accuracy ----------
    plt.figure(figsize=(10, 4))
    plt.plot(epochs, train_loss_list, label='Train Loss', color = 'orange')
    plt.plot(epochs, mae_list, label='Validation MAE', color = 'blue')
    plt.plot(epochs, rmse_list, label='Validation RMSE', color = 'green')
    plt.title('Train Loss vs Count Accuracy (MAE & RMSE)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss / Error')
    plt.legend()
    plt.tight_layout()
    plt.savefig("02. Count Accuracy.png")
    print("Plot saved as 2. Count Accuracy.png")

    # --------- Graph : Map Structure ----------
    plt.plot(epochs, train_loss_list, label='Train Loss', color = 'orange')
    plt.plot(epochs, ssim_list, label='Validation SSIM', color = 'blue')
    plt.plot(epochs, psnr_list, label='Validation PSNR', color = 'green')
    plt.title('Train Loss vs Map Structure (SSIM & PSNR)')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.tight_layout()
    plt.savefig("03. Map Structure .png") 
    print("Plot saved as 3. Map Structure.png")
    
    # --------- Graph: Evaluation Metrics ----------
    plt.figure(figsize=(10, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(epochs, mae_list, color = 'red')
    plt.title('MAE over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    
    
    plt.subplot(2, 2, 2)
    plt.plot(epochs, rmse_list, color='orange')
    plt.title('RMSE over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    
    plt.subplot(2, 2, 3)
    plt.plot(epochs, ssim_list, color='green')
    plt.title('SSIM over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('SSIM')

    plt.subplot(2, 2, 4)
    plt.plot(epochs, psnr_list, color='blue')
    plt.title('PSNR over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR')
    
    
    print("\n==== Final Accuracy Summary ====")
    print(f"Final Train Accuracy: {train_acc_list[-1]:.2f}%")
    print(f"Final Validation Accuracy: {val_acc_list[-1]:.2f}%")
    print(f"Best Validation Accuracy: {max(val_acc_list):.2f}%")
    
    plt.tight_layout()
    plt.savefig('04. Evaluation Metrics.png')
    print("Plot saved as 4. Evaluation Metrics.png")



    

def train(train_list, model, criterion, optimizer, epoch, device, loss_history):
    
    # For tracking average stats
    loss_meter = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    count_acc_meter = AverageMeter()
    
    # Prepare dataloader for training
    train_loader = torch.utils.data.DataLoader(
        dataset.Img_Density_Dataset(train_list,
            shuffle=True,                        
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
            ]), 
            train=True, 
            seen=model.seen,
            batch_size=args.batch_size,
            num_workers=args.workers),
        batch_size=args.batch_size)
    
    print('epoch %d, processed %d samples, lr %.10f' % (epoch, epoch * len(train_loader.dataset), args.lr))
    
    #print(f'Epoch {epoch}')
    #print(f'Total training images: {len(train_loader.dataset)}')
    #print(f'Batch size: {train_loader.batch_size}')
    #print(f'Total batches per epoch: {len(train_loader)}')
    #print(f'Learning rate: {args.lr:.10f}')

    model.train()
    end = time.time()
    
    for i, (img, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        
        img = img.to(device)
        
        target = target.type(torch.FloatTensor).to(device)
        
        if target.dim() == 3:
            target = target.unsqueeze(1)  # Add channel dimension
        
        output = model(img)
        

        #print(f"Batch {i}: img shape = {img.shape}, target shape = {target.shape}")
        
        # Losses: hybrid loss (MSE+SSIM) and count loss
        loss_a = criterion(output, target)  # HybridLoss (MSE+SSIM)
        loss_b = count_loss(output, target)  # Count loss
        loss = (ALPHA * loss_a + BETA * loss_b).mean()  # Weighted sum
        
        loss_history.append(loss.item())
        loss_meter.update(loss.item(), img.size(0))
        
        # Backward pass
        optimizer.zero_grad() # clear gradients
        loss.backward() # backward pass
        optimizer.step()
        
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  .format(epoch, i, len(train_loader), 
                          batch_time=batch_time,
                          data_time=data_time, 
                          loss=loss_meter))
            
        # Estimate count accuracy
        with torch.no_grad():
            pred_count = output.sum().item()
            gt_count = target.sum().item()
            if gt_count != 0:
                acc = 100 - (abs(pred_count - gt_count) / gt_count) * 100
            else:
                acc = 100.0 if pred_count == 0 else 0.0
            count_acc_meter.update(acc, img.size(0))
         
    return loss_meter.avg,count_acc_meter.avg

    



def eval_metrics(val_list, model, device):
    
    # Prepare validation dataloader
    test_loader = torch.utils.data.DataLoader(
        dataset.Img_Density_Dataset(val_list,
            shuffle=False,              
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
            ]), train=False),
        batch_size=args.batch_size)


    # Initialize metrics
    model.eval()
    mae = 0.0
    rmse = 0.0
    ssim_total = 0.0
    total_samples = 0
    psnr_total = 0.0
    val_acc_total = 0.0


    with torch.no_grad():
        for img, target in test_loader:
            img = img.to(device)
            target = target.type(torch.FloatTensor).to(device)
            
            if target.dim() == 3:
                target = target.unsqueeze(1)  # Add channel dimension
            
    
            output = model(img) # forward pass
            
            
            #print(f"output shape = {output.shape}, target shape (after resize) = {target.shape}")

            
            # MAE & RMSE 
            pred_count = output.sum()
            gt_count = target.sum()
            abs_err = abs(pred_count - gt_count).item()
            mae += abs_err
            rmse += abs_err**2
            
            #Count Accuracy
            if gt_count.item() != 0:
                acc = 100 - (abs_err / gt_count.item()) * 100
            else:
                acc = 100.0 if pred_count.item() == 0 else 0.0
            val_acc_total += acc


            # SSIM
            ssim_val = metrics.ssim(
                output if output.dim() == 4 else output.unsqueeze(0),
                target if target.dim() == 4 else target.unsqueeze(0),
                window_size=11
            ).mean().item()
            ssim_total += ssim_val

            # PSNR
            psnr_val = metrics.psnr(
                output if output.dim() == 4 else output.unsqueeze(0),
                target if target.dim() == 4 else target.unsqueeze(0),
                max_val=1.0  # adjust if maps aren't normalized to [0,1]
            ).mean().item()
            psnr_total += psnr_val
            
            total_samples += 1

    # Final averaged scores
    mae /= total_samples
    rmse = (rmse / total_samples) ** 0.5
    ssim_score = ssim_total / total_samples
    val_acc = val_acc_total / total_samples
    psnr_score = psnr_total / total_samples

    print(f' * MAE: {mae:.3f}, RMSE: {rmse:.3f}, SSIM: {ssim_score:.3f}, PSNR: {psnr_score:.3f}')
    return mae, rmse, ssim_score, psnr_score,val_acc
    




def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    args.lr = args.original_lr
    
    for i in range(len(args.steps)):
        scale = args.scales[i] if i < len(args.scales) else 1
        if epoch >= args.steps[i]:
            args.lr = args.lr * scale
            if epoch == args.steps[i]:
                break
        else:
            break
            
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__ == '__main__':
    main()
