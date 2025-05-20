import datasets
import modules
import os
import argparse
import torch.backends.cudnn as cudnn
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import pdb
import random

### NEW
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

parser = argparse.ArgumentParser()

# I/O PARAMS
parser.add_argument('--output', type=str, default='.', help='name of output directory')
parser.add_argument('--data', type=str, default='', help='which data to use')
parser.add_argument('--encoder', type=str, default='', choices=[
    'tres50_imagenet',
    'ctranspath',
    'phikon',
    'uni',
    'uni2',
    'virchow',
    'virchow2',
    'dinobloom-s',
    'dinobloom-g',
    'gigapath',
    'dinosmall',
    'dinobase'
], help='which encoder to use')
parser.add_argument('--method', type=str, default='', choices=[
    'AB-MIL',
    'AB-MIL_FC_small',
    'AB-MIL_FC_big',
    'CLAM_SB',
    'CLAM_MB',
    'transMIL',
    'Transformer',
    'Transformer2',
    'ViT',
    'DS-MIL',
    'VarMIL',
    'GTP',
    'PatchGCN',
    'DeepGraphConv',
    'ViT_MIL',
    'DTMIL',
    'LongNet_ViT'
], help='which aggregation method to use')
parser.add_argument('--kfold', default=0, type=int, choices=list(range(0,5)), help='which 0 t0 5?')
parser.add_argument('--num_classes', default=2, type=int, help='number of classes')

# OPTIMIZATION PARAMS
parser.add_argument('--momentum', default=0.9, type=float, help='momentum (default: 0.9)')
parser.add_argument("--lr", default=0.0005, type=float, help="""Learning rate at the end of linear warmup (highest LR used during training).""")
parser.add_argument('--lr_end', type=float, default=1e-6, help="""Target LR at the end of optimization. We use a cosine LR schedule with linear warmup.""")
parser.add_argument("--warmup_epochs", default=10, type=int, help="Number of epochs for the linear learning-rate warm up.")
parser.add_argument('--weight_decay', type=float, default=0.04, help="""Initial value of the weight decay.""")
parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the weight decay.""")
parser.add_argument('--nepochs', type=int, default=50, help='number of epochs (default: 40)')
parser.add_argument('--workers', default=10, type=int, help='number of data loading workers (default: 10)')
parser.add_argument('--random_seed', default=0, type=int, help='random seed')

def set_random_seed(seed_value):
    random.seed(seed_value)  # Python random module.
    np.random.seed(seed_value)  # Numpy module.
    torch.manual_seed(seed_value)  # Sets the seed for generating random numbers for CPU.
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)  # Sets the seed for generating random numbers on all GPUs.
        torch.cuda.manual_seed_all(seed_value)  # Sets the seed for generating random numbers on all GPUs.
        torch.backends.cudnn.deterministic = True  # Makes CUDA operations deterministic.
        torch.backends.cudnn.benchmark = False

def main():
    
    # Get user input
    global args
    args = parser.parse_args()

    if args.random_seed:
        set_random_seed(args.random_seed)
    
    ### NEW: Create a subdirectory for outputs based on the aggregation method
    args.save_dir = os.path.join(args.output, args.method)
    os.makedirs(args.save_dir, exist_ok=True)  # ensure directory exists
    
    # Now store the log file in this subdirectory
    log_path = os.path.join(args.save_dir, f'convergence_kfold{args.kfold}.csv')
    
    # Label dictionary (example for a certain dataset)
    label_dict = {'chipnegative': 0, 'chippositive': 1} 
    
    # Prepare datasets
    train_dset, val_dset, test_dset = datasets.get_datasets_kfold(
        kfold=args.kfold, data=args.data, encoder=args.encoder, label_dict=label_dict
    )
    train_loader = torch.utils.data.DataLoader(train_dset, batch_size=1, shuffle=True, num_workers=args.workers)
    val_loader = torch.utils.data.DataLoader(val_dset, batch_size=1, shuffle=False, num_workers=args.workers)
    
    # Dim of features based on encoder
    if args.encoder.startswith('tres50'):
        ndim = 1024
    elif args.encoder == 'ctranspath':
        ndim = 768
    elif args.encoder == 'phikon':
        ndim = 768
    elif args.encoder == 'uni':
        ndim = 1024
    elif args.encoder == 'uni2':
        ndim = 1536
    elif args.encoder == 'virchow':
        ndim = 2560
    elif args.encoder == 'virchow2':
        ndim = 2560
    elif args.encoder == 'dinobloom-s':
        ndim = 384
    elif args.encoder == 'dinobloom-g':
        ndim = 1536
    elif args.encoder == 'gigapath':
        ndim = 1536
    elif args.encoder.startswith('dinosmall'):
        ndim = 384
    elif args.encoder.startswith('dinobase'):
        ndim = 768
    
    # Get model
    model = modules.get_aggregator(method=args.method, ndim=ndim, n_classes=args.num_classes)
    model.cuda()
    
    # Set loss
    criterion = nn.CrossEntropyLoss().cuda()
    
    # Set optimizer
    params_groups = get_params_groups(model)
    optimizer = optim.AdamW(params_groups)
    
    # Set schedulers
    lr_schedule = cosine_scheduler(
        args.lr,
        args.lr_end,
        args.nepochs,
        len(train_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.nepochs,
        len(train_loader),
    )
    cudnn.benchmark = True
    
    # Initialize logs
    with open(log_path, 'w') as fconv:
        fconv.write('epoch,metric,value\n')
    
    ### Track the best AUC, best F1, and best Accuracy
    best_auc = 0.0
    best_f1 = 0.0
    best_acc = 0.0
    
    # Main training loop
    for epoch in range(args.nepochs+1):
        
        # Training
        if epoch > 0:
            loss = train(epoch, train_loader, model, criterion, optimizer, lr_schedule, wd_schedule)
            print('Training\tEpoch: [{}/{}]\tLoss: {}'.format(epoch, args.nepochs, loss))
            with open(log_path, 'a') as fconv:
                fconv.write(f"{epoch},loss,{loss}\n")
        
        # Validation
        probs = test(epoch, val_loader, model, args)
        
        # Convert the ground-truth to numeric array
        y_true = val_loader.dataset.df['target'].map(label_dict).values
        # AUC score
        pos_probs = probs[:, 1]  # since class 1 is the "positive" class
        auc = roc_auc_score(y_true, pos_probs)
        
        # F1 score (macro); get predicted labels
        y_pred = np.argmax(probs, axis=1)
        f1 = f1_score(y_true, y_pred, average='binary')

        # Accuracy
        acc = accuracy_score(y_true, y_pred)
        
        # Check if the current model is best by AUC
        if auc > best_auc:
            print(f"New best AUC model found at epoch {epoch} with AUC: {auc}")
            best_auc = auc
            # Save checkpoint
            obj = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'auc': auc,
                'f1': f1,
                'acc': acc,
                'optimizer': optimizer.state_dict()
            }
            ### Save best checkpoint (by AUC)
            torch.save(obj, os.path.join(args.save_dir, f'checkpoint_best_auc_kfold{args.kfold}.pth'))
        
        # Check if the current model is best by F1
        if f1 > best_f1:
            print(f"New best F1 model found at epoch {epoch} with F1: {f1}")
            best_f1 = f1
            # Save checkpoint
            obj = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'auc': auc,
                'f1': f1,
                'acc': acc,
                'optimizer': optimizer.state_dict()
            }
            ### Save best checkpoint (by F1)
            torch.save(obj, os.path.join(args.save_dir, f'checkpoint_best_f1_kfold{args.kfold}.pth'))
        
        # Check if the current model is best by Accuracy
        if acc > best_acc:
            print(f"New best Accuracy model found at epoch {epoch} with Accuracy: {acc}")
            best_acc = acc
            # Save checkpoint
            obj = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'auc': auc,
                'f1': f1,
                'acc': acc,
                'optimizer': optimizer.state_dict()
            }
            torch.save(obj, os.path.join(args.save_dir, f'checkpoint_best_acc_kfold{args.kfold}.pth'))

        # Print stats
        print('Validation\tEpoch: [{}/{}]\tAUC: {:.4f}\tF1: {:.4f}\tAcc: {:.4f}'.format(
            epoch, args.nepochs, auc, f1, acc))
        with open(log_path, 'a') as fconv:
            fconv.write(f"{epoch},auc,{auc}\n")
            fconv.write(f"{epoch},f1,{f1}\n")
            fconv.write(f"{epoch},acc,{acc}\n")
        
        # Always save a 'latest' checkpoint if you want
        obj = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'auc': auc,
            'f1': f1,
            'acc': acc,
            'optimizer': optimizer.state_dict()
        }
        torch.save(obj, os.path.join(args.save_dir, f'checkpoint_latest_kfold{args.kfold}.pth'))

def test(run, loader, model, args):
    # Set model in test mode
    model.eval()
    # Initialize probability vector
    probs = torch.FloatTensor(len(loader), args.num_classes).cuda()
    # Loop through batches
    with torch.no_grad():
        for i, (input, _) in enumerate(loader):
            print('Inference\tEpoch: [{}/{}]\tBatch: [{}/{}]'.format(run, args.nepochs, i+1, len(loader)))
            # Copy batch to GPU
            feat = input.float().squeeze(0).cuda()
            results_dict = model(feat)
            logits, Y_prob, Y_hat = (results_dict[key] for key in ['logits', 'Y_prob', 'Y_hat'])
            # If Y_prob is Nx1 for some reason, squeeze
            if Y_prob.dim() > 1 and Y_prob.size(1) == 1:
                Y_prob = Y_prob.squeeze(1)
            probs[i] = Y_prob.detach()
    return probs.cpu().numpy()

def train(run, loader, model, criterion, optimizer, lr_schedule, wd_schedule):
    model.train()
    running_loss = 0.
    for i, (input, target) in enumerate(loader):
        # Update weight decay and learning rate
        it = len(loader) * (run-1) + i  # global training iteration
        for j, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if j == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]
        
        feat = input.float().squeeze(0).cuda()
        target = target.long().cuda()
        
        # Forward pass
        results_dict = model(feat)
        logits = results_dict['logits']
        loss = criterion(logits, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        print('Training\tEpoch: [{}/{}]\tBatch: [{}/{}]\tLoss: {}'.format(
            run, args.nepochs, i+1, len(loader), loss.item()))
    return running_loss / len(loader)

def get_params_groups(model):
    regularized = []
    not_regularized = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # we do not regularize biases nor Norm parameters
        if name.endswith(".bias") or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)
    return [{'params': regularized}, {'params': not_regularized, 'weight_decay': 0.}]

def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule

if __name__ == '__main__':
    main()
