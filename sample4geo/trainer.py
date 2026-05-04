import time
import torch
from tqdm import tqdm
from .utils import AverageMeter
from torch.cuda.amp import autocast
import torch.nn.functional as F
import torch.nn as nn
from sample4geo.cal_loss import cal_kl_loss, cal_loss, cal_triplet_loss
def train(train_config, model, dataloader, loss_function, optimizer, scheduler=None, scaler=None):

    # set model train mode
    model.train()
    
    losses = AverageMeter()
    
    # wait before starting progress bar
    time.sleep(0.1)
    
    # Zero gradients for first step
    optimizer.zero_grad(set_to_none=True)
    
    step = 1
    
    if train_config.verbose:
        bar = tqdm(dataloader, total=len(dataloader))
    else:
        bar = dataloader

    criterion = nn.CrossEntropyLoss()

    # for loop over one epoch
    for query, reference, ids, labels in bar:
        if scaler:
            with autocast():
            
                # data (batches) to device   
                query = query.to(train_config.device)
                reference = reference.to(train_config.device)
                labels = labels.to(train_config.device)
            
                # Forward pass
                outputs1, outputs2 = model(query, reference)

                features1, fg1, bg1 = outputs1[-2], outputs1[0], outputs1[-1]
                features2, fg2, bg2 = outputs2[-2], outputs2[0], outputs1[-1]
                fg1 = fg1.mean(dim=1)  # [8, 1024, 1369] → [8, 1369]
                fg2 = fg2.mean(dim=1) 
                # 处理 bg1, bg2 （可能是逐位置分类）
                bg1 = bg1.mean(dim=1)  # [4, 768, 37, 37] → [4, 37, 37]
                bg1 = bg1.flatten(1)   # [4, 37, 37] → [4, 1369]

                bg2 = bg2.mean(dim=1)  # [4, 768, 37, 37] → [4, 37, 37]
                bg2 = bg2.flatten(1)   # [4, 37, 37] → [4, 1369]
                # print(labels.shape)  # 应该是 [batch_size]
                if torch.cuda.device_count() > 1 and len(train_config.gpu_ids) > 1: 
                    loss = loss_function(features1, features2, model.module.logit_scale.exp())
                    # loss2 = loss_function(fg, fg, model.module.logit_scale.exp())
                    loss3 = cal_loss(bg1, labels, criterion) + cal_loss(bg2, labels, criterion)
                    loss2 = cal_loss(fg1, labels, criterion) + cal_loss(fg2, labels, criterion)


                else:
                    loss = loss_function(features1, features2, model.module.logit_scale.exp()) 
                # lossall = loss + 0.1 * loss2 + 0.1 * loss3 #95.1406 
                # lossall = 0.8 * loss + 0.1 * loss2 + 0.1 * loss3  95.04
                # lossall = 1.0 * loss + 0.15 * loss2 + 0.15 * loss3 
                lossall = loss + 1 * loss2 + 0.7 * loss3

                losses.update(lossall.item())
                
                  
            scaler.scale(lossall).backward()
            
            # Gradient clipping 
            if train_config.clip_grad:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_value_(model.parameters(), train_config.clip_grad) 
            
            # Update model parameters (weights)
            scaler.step(optimizer)
            scaler.update()

            # Zero gradients for next step
            optimizer.zero_grad()
            
            # Scheduler
            if train_config.scheduler == "polynomial" or train_config.scheduler == "cosine" or train_config.scheduler ==  "constant":
                scheduler.step()
   
        else:
        
            # data (batches) to device   
            query = query.to(train_config.device)
            reference = reference.to(train_config.device)

            # Forward pass
            features1, features2 = model(query, reference)
            if torch.cuda.device_count() > 1 and len(train_config.gpu_ids) > 1: 
                loss = loss_function(features1, features2, model.module.logit_scale.exp())
            else:
                loss = loss_function(features1, features2, model.logit_scale.exp()) 
            losses.update(loss.item())

            # Calculate gradient using backward pass
            loss.backward()
            
            # Gradient clipping 
            if train_config.clip_grad:
                torch.nn.utils.clip_grad_value_(model.parameters(), train_config.clip_grad)                  
            
            # Update model parameters (weights)
            optimizer.step()
            # Zero gradients for next step
            optimizer.zero_grad()
            
            # Scheduler
            if train_config.scheduler == "polynomial" or train_config.scheduler == "cosine" or train_config.scheduler ==  "constant":
                scheduler.step()
        
        
        
        if train_config.verbose:
            
            monitor = {"loss": "{:.4f}".format(loss.item()),
                       "loss_avg": "{:.4f}".format(losses.avg),
                       "lr" : "{:.6f}".format(optimizer.param_groups[0]['lr'])}
            
            bar.set_postfix(ordered_dict=monitor)
        
        step += 1

    if train_config.verbose:
        bar.close()

    return losses.avg


def predict(train_config, model, dataloader):
    
    model.eval()
    
    # wait before starting progress bar
    time.sleep(0.1)
    
    if train_config.verbose:
        bar = tqdm(dataloader, total=len(dataloader))
    else:
        bar = dataloader
        
    img_features_list = []
    
    ids_list = []
    with torch.no_grad():
        
        for img, ids in bar:
        
            ids_list.append(ids)
            
            with autocast():
         
                img = img.to(train_config.device)
                img_feature = model(img)[0]

                # normalize is calculated in fp32
                if train_config.normalize_features:
                    img_feature = F.normalize(img_feature, dim=-1)
            
            # save features in fp32 for sim calculation
            img_features_list.append(img_feature.to(torch.float32))
      
        # keep Features on GPU
        img_features = torch.cat(img_features_list, dim=0) 
        ids_list = torch.cat(ids_list, dim=0).to(train_config.device)
        
    if train_config.verbose:
        bar.close()
        
    return img_features, ids_list