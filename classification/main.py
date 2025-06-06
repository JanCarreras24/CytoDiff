import os
import sys
import json
import time
import math
import random
import datetime
import traceback
from pathlib import Path
from os.path import join as ospj
import wandb
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F

from torchvision.transforms import v2

from utils import (
    fix_random_seeds,
    cosine_scheduler,
    MetricLogger,
)

from config import get_args
from data import get_data_loader, get_synth_train_data_loader
from models.clip import CLIP
from models.resnet50 import ResNet50
from util_data import SUBSET_NAMES, PROMPTS_BY_CLASS
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import label_binarize




def load_data_loader(args):
    train_loader, val_loader, test_loader = get_data_loader(
        dataroot=args.dataroot,  # Path 
        dataset_selection=args.dataset_selection,  # Dataset
        bs=args.batch_size,
        eval_bs=args.batch_size_eval,
        is_rand_aug=args.is_rand_aug,
        model_type=args.model_type,
        fold=args.fold,  # Fold para k-fold cross-validation
        is_hsv=args.is_hsv,  # Control de HSV
        is_hed=args.is_hed,  # Control de HED
    )
    return train_loader, val_loader, test_loader



def load_synth_train_data_loader(args):
    synth_train_loader = get_synth_train_data_loader(
        dataroot=args.dataroot,  # Path
        dataset_selection=args.dataset_selection,
        bs=args.batch_size,
        is_rand_aug=args.is_rand_aug,
        model_type=args.model_type,
        fold=args.fold,
        is_hsv=args.is_hsv,
        is_hed=args.is_hed,
    )
    return synth_train_loader


def main(args):
    args.n_classes = len(SUBSET_NAMES[args.dataset_selection])

    os.makedirs(args.output_dir, exist_ok=True)

    fix_random_seeds(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    cudnn.benchmark = True

    # ==================================================
    # Data loader
    # ==================================================
    train_loader, val_loader, test_loader = load_data_loader(args)
    if args.is_synth_train:
        print("Loading synthetic training data...")
        train_loader = load_synth_train_data_loader(args)

        
    # ==================================================
    # Model and optimizer
    # ==================================================
    if args.model_type == "clip":

        model = CLIP(
            dataset=args.dataset_selection,
            is_lora_image=args.is_lora_image,
            is_lora_text=args.is_lora_text,
            clip_download_dir=args.clip_download_dir,
            clip_version=args.clip_version,
        )
        params_groups = model.learnable_params()
    elif args.model_type == "resnet50": 
        model = ResNet50(n_classes=args.n_classes)
        params_groups = model.parameters()

    model = model.cuda()

    criterion = nn.CrossEntropyLoss().cuda()

    # CutMix and MixUp augmentation
    if args.is_mix_aug:
        cutmix = v2.CutMix(num_classes=args.n_classes)
        mixup = v2.MixUp(num_classes=args.n_classes)
        cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])
    else:
        cutmix_or_mixup = None

    scheduler = None
    optimizer = torch.optim.AdamW(
        params_groups, lr=args.lr, weight_decay=args.wd,
    )
    args.lr_schedule = cosine_scheduler(
        args.lr,
        args.min_lr,
        args.epochs,
        len(train_loader),
        warmup_epochs=args.warmup_epochs,
        start_warmup_value=args.min_lr,
    )

    fp16_scaler = None
    if args.use_fp16:
        # mixed precision training
        fp16_scaler = torch.cuda.amp.GradScaler()

    # ==================================================
    # Loading previous checkpoint & initializing tensorboard
    # ==================================================

    if args.log == 'wandb':
        assert wandb is not None, "Wandb not installed, please install it or run without wandb"
        _ = os.system('wandb login {}'.format(args.wandb_key))
        os.environ['WANDB_API_KEY'] = args.wandb_key
        wandb.init(
            project=args.wandb_project, 
            group=args.wandb_group, 
            name=args.wandb_group,
            #settings=wandb.Settings(start_method='fork'),
            settings=wandb.Settings(start_method='spawn'), # spawn, thread
            config=vars(args)
        )
        args.wandb_url = wandb.run.get_url()
    elif args.log == 'tensorboard':
        from torch.utils.tensorboard import SummaryWriter
        tb_dir = os.path.join(args.output_dir, "tb-{}".format(args.local_rank))
        Path(tb_dir).mkdir(parents=True, exist_ok=True)
        tb_writer = SummaryWriter(tb_dir, flush_secs=30)

    # ==================================================
    # Training
    # ==================================================
    print("=> Training starts ...")
    start_time = time.time()

    best_stats = {}
    best_top1 = 0.

    for epoch in range(0, args.epochs):
        # Entrenamiento
        train_stats, best_stats, best_top1 = train_one_epoch(
            model, criterion, train_loader, optimizer, scheduler, epoch, fp16_scaler, cutmix_or_mixup, args,
            best_stats, best_top1
        )

        # Evaluación en el conjunto de validación
        val_stats = eval(
            model, criterion, val_loader, epoch, fp16_scaler, args, prefix="val")
        print(f"Validation stats for epoch {epoch}: {val_stats}")
        #print(f"Val images: {len(val_loader.dataset)}")

                
        '''
        # Guardar el mejor modelo basado en la métrica de validación
        if val_stats["val/top1"] > best_top1:
            best_top1 = val_stats["val/top1"]
            best_stats = val_stats
            save_model(args, model, optimizer, epoch, fp16_scaler, "best_checkpoint.pth")

        if epoch + 1 == args.epochs:
            val_stats['val/best_top1'] = best_stats["val/top1"]
            val_stats['val/best_loss'] = best_stats["val/loss"]

        if args.log == 'wandb':
            train_stats.update({"epoch": epoch})
            wandb.log(train_stats)
            wandb.log(val_stats)        
        '''


    # ==================================================
    # Final evaluation on test set
    # ==================================================
    print("=> Evaluating on test set...")
    print(f"Test images: {len(test_loader.dataset)}")

    test_stats = eval(
        model, criterion, test_loader, args.epochs, fp16_scaler, args, prefix="test")

    print(f"Test stats: {test_stats}")

    #total_time = time.time() - start_time
    #total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    #print("Training time {}".format(total_time_str))


def train_one_epoch(
    model, criterion, data_loader, optimizer, scheduler, epoch, fp16_scaler, cutmix_or_mixup, args,
    best_stats, best_top1,
):
    metric_logger = MetricLogger(delimiter="  ")
    header = "Epoch: [{}/{}]".format(epoch, args.epochs)

    model.train()

    for it, batch in enumerate(data_loader):
        if args.is_synth_train and args.is_pooled_fewshot:
            image, label, is_real = batch
        else:
            image, label = batch

        label_origin = label
        label_origin = label_origin.cuda(non_blocking=True)


        # apply CutMix and MixUp augmentation
        if args.is_mix_aug:
            p = random.random()
            if p >= 0.2:
                pass
            else:
                if args.is_synth_train and args.is_pooled_fewshot:
                    new_image = torch.zeros_like(image)
                    new_label = torch.stack([torch.zeros_like(label)] * args.n_classes, dim=1).mul(1.0)

                    image_real, label_real = image[is_real==1], label[is_real==1]
                    image_synth, label_synth = image[is_real==0], label[is_real==0]

                    image_real, label_real = cutmix_or_mixup(image_real, label_real)
                    image_synth, label_synth = cutmix_or_mixup(image_synth, label_synth)

                    new_image[is_real==1] = image_real
                    new_image[is_real==0] = image_synth
                    new_label[is_real==1] = label_real
                    new_label[is_real==0] = label_synth

                    image = new_image
                    label = new_label

                else:
                    image, label = cutmix_or_mixup(image, label)

            

        it = len(data_loader) * epoch + it  # global training iteration

        image = image.squeeze(1).to(torch.float16).cuda(non_blocking=True)
        label = label.cuda(non_blocking=True)

        # update weight decay and learning rate according to their schedule
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = args.lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = args.wd
                

        # forward pass
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            logit = model(image)



            if args.is_synth_train and args.is_pooled_fewshot:
                mask_real = (is_real == 1)
                mask_synth = (is_real == 0)
                #print("  mask_real sum:", mask_real.sum().item(), "mask_synth sum:", mask_synth.sum().item())

                n_real = mask_real.sum().item()
                n_synth = mask_synth.sum().item()
                n_total = n_real + n_synth

                if n_total == 0:
                    print("[ERROR] No samples in batch!")
                    sys.exit(1)

                loss = 0.0

                if n_real > 0:
                    loss_real = criterion(logit[mask_real], label[mask_real])
                    weighted_loss_real = args.lambda_1 * (n_real / n_total) * loss_real
                    loss += weighted_loss_real
                else:
                    pass

                if n_synth > 0:
                    loss_synth = criterion(logit[mask_synth], label[mask_synth])
                    weighted_loss_synth = (1 - args.lambda_1) * (n_synth / n_total) * loss_synth
                    loss += weighted_loss_synth
                else:
                    pass

                #print("  loss_real:", loss_real.item() if n_real > 0 else "-", 
                    #"loss_synth:", loss_synth.item() if n_synth > 0 else "-", 
                    #"loss:", loss.item())

            else:
                loss = criterion(logit, label)

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            sys.exit(1)


        # parameter update
        optimizer.zero_grad()
        if fp16_scaler is None:
            loss.backward()
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # logging
        with torch.no_grad():
            acc1, acc5 = get_accuracy(logit.detach(), label_origin, topk=(1, 5))
            metric_logger.update(top1=acc1.item())
            metric_logger.update(loss=loss.item())
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
            metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])

        if scheduler is not None:
            scheduler.step()

    metric_logger.synchronize_between_processes()
    return {"train/{}".format(k): meter.global_avg for k, meter in metric_logger.meters.items()}, best_stats, best_top1


@torch.no_grad()

def eval(model, criterion, data_loader, epoch, fp16_scaler, args, prefix="test"):

    metric_logger = MetricLogger(delimiter="  ")
    header = "Epoch: [{}/{}]".format(epoch, args.epochs)

    is_last = epoch + 1 == args.epochs

    targets = []
    outputs = []

    model.eval()

    #probs = torch.FloatTensor(len(data_loader), args.n_classes).cuda()
    for it, (image, label) in enumerate(
        #metric_logger.log_every(data_loader, 100, header)
        data_loader
    ):
        image = image.cuda(non_blocking=True)
        label = label.cuda(non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            output = model(image, phase="eval")
            loss = criterion(output, label)
            #probs[it] = output.softmax(dim=1).detach() #help rao

        acc1, acc5 = get_accuracy(output, label, topk=(1, 5))

        # record logs
        metric_logger.update(loss=loss.item())
        metric_logger.update(top1=acc1.item())
        metric_logger.update(top5=acc5.item())

        targets.append(label.cpu())
        outputs.append(output.cpu())  # logits (sin softmax aún)

    metric_logger.synchronize_between_processes()
    #print(f"Averaged {prefix} stats:", metric_logger)

    stat_dict = {f"{prefix}/{k}": meter.global_avg for k, meter in metric_logger.meters.items()}

    targets = torch.cat(targets).detach().cpu().numpy()
    outputs = torch.cat(outputs)

    # Obtain probabilities for auc
    probs = F.softmax(outputs, dim=1).detach().cpu().numpy()

    # Obtain predictions
    preds = probs.argmax(axis=1)

    # F1-scores
    f1_macro = f1_score(targets, preds, average="macro")
    f1_micro = f1_score(targets, preds, average="micro")
    f1_weighted = f1_score(targets, preds, average="weighted")

    stat_dict[f"{prefix}/f1_macro"] = f1_macro
    stat_dict[f"{prefix}/f1_micro"] = f1_micro
    stat_dict[f"{prefix}/f1_weighted"] = f1_weighted

    # AUC
    unique_targets = np.unique(targets)

    if len(unique_targets) == args.n_classes:
        try:
            y_true_bin = label_binarize(targets, classes=np.arange(args.n_classes))
            auc_macro = roc_auc_score(y_true_bin, probs, average="macro", multi_class="ovr")
            #auc_micro = roc_auc_score(y_true_bin, probs, average="micro", multi_class="ovr")
            auc_weighted = roc_auc_score(y_true_bin, probs, average="weighted", multi_class="ovr")

        except Exception as e:
            print("ERROR during roc_auc_score:", e)
    else:
        print(f"[WARN] Not all classes present in target: {unique_targets.tolist()}")

    stat_dict[f"{prefix}/auc_macro"] = auc_macro
    #stat_dict[f"{prefix}/auc_micro"] = auc_micro
    stat_dict[f"{prefix}/auc_weighted"] = auc_weighted
    
    
    if prefix == "test":
        # Confusion matrix
        conf_matrix = confusion_matrix(targets, preds)
        print(f"{prefix.capitalize()} Confusion Matrix:\n{conf_matrix}")

        # Calculate per class accuracy
        targets_acc = torch.tensor(targets)
        probs_acc = torch.tensor(probs)
        acc_per_class = [
            get_accuracy(probs_acc[targets_acc == cls_idx], targets_acc[targets_acc == cls_idx], topk=(1,))[0].item()
            for cls_idx in range(args.n_classes)
        ]
        for cls_idx, acc in enumerate(acc_per_class):
            print(f"{SUBSET_NAMES[args.dataset_selection][cls_idx]} [{cls_idx}]: {acc}")
            #stat_dict[f"{prefix}/{SUBSET_NAMES[args.dataset_selection][cls_idx]}_cls-acc"] = acc

    return stat_dict


def get_accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:k].reshape(-1).float().sum(0) * 100.0 / batch_size for k in topk]



def save_model(args, model, optimizer, epoch, fp16_scaler, file_name):
    state_dict = model.state_dict()
    save_dict = {
        "model": state_dict,
        "optimizer": optimizer.state_dict(),
        "epoch": epoch + 1,
        "args": args,
    }
    if fp16_scaler is not None:
        save_dict["fp16_scaler"] = fp16_scaler.state_dict()
    torch.save(save_dict, os.path.join(args.output_dir, file_name))


if __name__ == "__main__":
    try:
        print("Inicio del script...")
        args = get_args()
        print(f"Argumentos recibidos: {args}")
        main(args)
        print("Fin del script.")
    except Exception as e:
        print(f"Error durante la ejecución: {e}")
