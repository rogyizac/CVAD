import os
import time
import copy
import logging
import numpy as np

import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

import torchvision.utils as vutils
from torchvision.utils import save_image

from evaluate import *
from utils.bert_utils import init_bert_model

# os.environ["TOKENIZERS_PARALLELISM"] = "false"

def load_ckpt(checkpoint_fpath, model, optimizer):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['model_state_dict'])
    epochs_run = checkpoint['epoch']
    return model, epochs_run

def save_checkpoint(state, filename):
    """Save checkpoint if a new best is achieved"""
    print ("=> Saving a new best")
    torch.save(state, filename)  # save checkpoint
    

def train_all(netG, netD, efnet, imgSize, variational_beta, cvae_batch_size, optimizerG, optimizerD, optimizerE, recon_loss, cls_loss, dataset, train_loader, val_loader, test_loader, train_img_caption_dataloader, val_img_caption_dataloader, test_img_caption_dataloader, Gepoch, Gepochs_run, Depoch, Depochs_run, Eepoch, Eepochs_run, channel, device, evaluation_flag, normal_class, effusion_model_flag):
    
    logger = logging.getLogger()
    scaler = GradScaler()

    best_loss = np.inf
    best_loss2 = np.inf
    best_loss3 = np.inf
    
    schedulerG = ReduceLROnPlateau(optimizerG, mode='min', factor=0.5, patience=10, verbose=True)
    schedulerD = ReduceLROnPlateau(optimizerD, mode='min', factor=0.5, patience=5, verbose=True)
    schedulerE = ReduceLROnPlateau(optimizerE, mode='min', factor=0.5, patience=10, verbose=True)

    ################################################################################
    # train CVAE
    ################################################################################
    netG = netG.to(f'cuda:{device}')
    netG = nn.SyncBatchNorm.convert_sync_batchnorm(netG)
    print(f"DDP NetG Starts on {device}")
    netG = DDP(netG, device_ids=[device], find_unused_parameters=True)
    print(f"DDP NetG Ends on {device}")
    
    for epoch in range(Gepoch):
        loss = []
        netG.train()
        train_loader.sampler.set_epoch(epoch)
        current_lr = optimizerG.param_groups[0]['lr']
        # if device == 0:
        #     cvae_writer.add_scalar('cvae lr', float(current_lr), epoch)
        logger.info(f"Epoch {epoch}, Current LR: {current_lr}")
        for i, (images,_) in enumerate(train_loader):
            optimizerG.zero_grad()
            images = images.to(device)
            with autocast(dtype=torch.float16): # bfloat16
                recon_x, mu, logvar, mu2, logvar2 = netG(images)
                L_dec_vae = recon_loss(recon_x, images, mu, logvar, mu2, logvar2, variational_beta, imgSize, channel, cvae_batch_size)
                
            scaler.scale(L_dec_vae).backward()
            scaler.step(optimizerG)
            scaler.update()
            loss.append(L_dec_vae.item())
        
        if device == 0:
            vutils.save_image(images,
                            './logs/'+dataset + '/' + str(normal_class) + '/real_samples_'+dataset+'.png',normalize=True)
            vutils.save_image(recon_x.data.view(-1,channel,imgSize,imgSize),
                            './logs/'+dataset+ '/' + str(normal_class) + '/fake_samples_'+dataset+'.png',normalize=True)
            img_grid = vutils.make_grid(images)
            # cvae_writer.add_image('original images', img_grid)
            img_grid = vutils.make_grid(recon_x.data.view(-1,channel,imgSize,imgSize))
            # cvae_writer.add_image('reconstructed images', img_grid)

        logger.info("Epoch:%d Trainloss: %.8f"%(epoch, np.mean(loss)))
        # if device == 0:
            # cvae_writer.add_scalar('training loss', np.mean(loss), epoch)
    
        loss = []
        netG.eval()
        val_loader.sampler.set_epoch(epoch)
        for i, (images,_) in enumerate(val_loader):
            images = images.to(device)
            with autocast(dtype=torch.float16):
                recon_x, mu, logvar, mu2, logvar2 = netG(images)
                L_dec_vae = recon_loss(recon_x, images, mu, logvar, mu2, logvar2, variational_beta, imgSize, channel, cvae_batch_size)
            loss.append(L_dec_vae.item())
        schedulerG.step(np.mean(loss))
        
        logger.info("Epoch:%d   Valloss: %.8f"%(epoch, np.mean(loss)))
        # if device == 0:
        #     cvae_writer.add_scalar('validation loss', np.mean(loss), epoch)
        
        if (device == 0) and (np.mean(loss)<best_loss):
            best_loss = np.mean(loss)
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': netG.module.state_dict(),
                    'optimizer_state_dict': optimizerG.state_dict(),
                    }, "../Data/risaac6/weights/"+dataset+"/netG_"+dataset+".pth.tar")
    netG.eval()      
    # if evaluation_flag:
    #     cvae_evaluate(netG, recon_loss, test_loader, device, variational_beta, imgSize, channel, cvae_batch_size)
        

    ###############################################################################
    # train Discriminator
    ################################################################################    
        
    logger.info("--------CLS--------")
    netD = netD.to(f'cuda:{device}')
    netD = nn.SyncBatchNorm.convert_sync_batchnorm(netD)
    netD = DDP(netD, device_ids=[device])
    cls_loss = torch.nn.BCELoss()
    netG.eval()
    for epoch in range(Depoch):
        loss = []
        train_loader.sampler.set_epoch(epoch)
        current_lr = optimizerD.param_groups[0]['lr']
        netD.train()
        # if device == 0:
        #     cls_writer.add_scalar('cls lr', float(current_lr), epoch)
        logger.info(f"Epoch {epoch}, Current LR: {current_lr}")
        for i, (images, targets) in enumerate(train_loader):
            images = images.to(device)
            targets = targets.to(device)
            optimizerD.zero_grad()
            recon_x, mu, logvar, mu2, logvar2 = netG(images)
            preds = netD(images)
            preds2 = netD(recon_x)
            
            L_dec_vae = cls_loss(torch.squeeze(preds, dim=1),targets.float())
            L_dec_vae += cls_loss(torch.squeeze(preds2, dim=1),1.0-targets)
            L_dec_vae.backward()
            optimizerD.step()      
            loss.append(L_dec_vae.item())
            
   
        logger.info("Epoch:%d Trainloss: %.8f"%(epoch, np.mean(loss)))
        # if device == 0:
        #     cls_writer.add_scalar('training loss', np.mean(loss), epoch)
        loss = []
        netD.eval()
        val_loader.sampler.set_epoch(epoch)
        for i, (images, targets) in enumerate(val_loader):
            images = images.to(device)
            targets = targets.to(device)
            recon_x, mu, logvar, mu2, logvar2 = netG(images)
            preds = netD(images)
            preds2 = netD(recon_x)

            L_dec_vae = cls_loss(torch.squeeze(preds, dim=1),targets.float())
            L_dec_vae += cls_loss(torch.squeeze(preds2, dim=1),1.0-targets)
            loss.append(L_dec_vae.item())

        schedulerD.step(np.mean(loss))
        logger.info("Epoch:%d   Valloss: %.8f"%(epoch, np.mean(loss)))
        # if device == 0:
        #     cls_writer.add_scalar('validation loss', np.mean(loss), epoch)
        if (device == 0) and (np.mean(loss)<best_loss2):
            best_loss2 = np.mean(loss)
            torch.save({
                'epoch': epoch,
                'model_state_dict': netD.module.state_dict(),
                'optimizer_state_dict': optimizerD.state_dict(),
                }, "../Data/risaac6/weights/"+dataset+"/netD_"+dataset+".pth.tar")
        
    netD.eval()
    # if evaluation_flag:
    #     cvad_evaluate(netG, netD, recon_loss, cls_loss, test_loader, device, variational_beta, imgSize, channel, cvae_batch_size, 0, normal_class)

    ###############################################################################
    # train Early Fusion Net
    ################################################################################    
    
    
    if effusion_model_flag:
        logger.info("--------EFnet--------")
        efnet = efnet.to(f'cuda:{device}')
        efnet = nn.SyncBatchNorm.convert_sync_batchnorm(efnet)
        efnet = DDP(efnet, device_ids = [device])
        efnet_loss = nn.BCELoss()        

        netG.eval()
        for epoch in range(Eepoch):
            loss = []
            train_img_caption_dataloader.sampler.set_epoch(epoch)
            current_lr = optimizerE.param_groups[0]['lr']
            logger.info(f"Epoch {epoch}, Current LR: {current_lr}")
            efnet.train()
            for (i, batch) in enumerate(train_img_caption_dataloader):
                # unpack inputs
                batch['img'] = batch['img'].to(device)
                batch['embeddings'] = batch['embeddings'].to(device)
                batch['caption_label'] = batch['caption_label'].to(device)
                images = batch['img']
                embeddings = batch['embeddings']
                labels = batch['caption_label']
                optimizerE.zero_grad()
                # image reconstruction output
                with torch.no_grad():
                    recon_x1, mu1, logvar1, mu21, logvar21 = netG(images)
                    recon_x2, mu2, logvar2, mu22, logvar22 = netG(recon_x1)
                # Create inputs
                inputs1 = {'image_emb':mu1, 'text_emb':embeddings.squeeze(1)}
                inputs2 = {'image_emb':mu2, 'text_emb':embeddings.squeeze(1)}
                # early fusion network pass for image
                outputs1 = efnet(**inputs1)
                outputs2 = efnet(**inputs2)
                # Calculate loss
                loss_value = efnet_loss(torch.squeeze(outputs1, dim=1), labels.float())
                loss_value += efnet_loss(torch.squeeze(outputs2, dim=1), 1.0 - labels)
                # back prop
                loss_value.backward()
                optimizerE.step()
                loss.append(loss_value.item())
                # print((i+1)*len(batch['img']))
                if (i+1)*len(batch['img']) > 50000:
                    # print('train break')
                    break
            # Compute mean loss or handle empty loss list
            if loss:
                epoch_loss = np.mean(loss)
            else:
                logger.info("Loss list is empty.")
                epoch_loss = 0

            logger.info("Epoch:%d Trainloss: %.8f"%(epoch, epoch_loss))

            loss = []
            efnet.eval()
            val_img_caption_dataloader.sampler.set_epoch(epoch)
            with torch.no_grad():
                for (i, batch) in enumerate(val_img_caption_dataloader):
                    # parse inputs
                    batch['img'] = batch['img'].to(device)
                    batch['embeddings'] = batch['embeddings'].to(device)
                    batch['caption_label'] = batch['caption_label'].to(device)
                    images = batch['img']
                    embeddings = batch['embeddings']
                    labels = batch['caption_label']

                    recon_x1, mu1, logvar1, mu21, logvar21 = netG(images)
                    recon_x2, mu2, logvar2, mu22, logvar22 = netG(recon_x1)
                    
                    inputs1 = {'image_emb':mu1, 'text_emb':embeddings.squeeze(1)}
                    inputs2 = {'image_emb':mu2, 'text_emb':embeddings.squeeze(1)}
                    
                    outputs1 = efnet(**inputs1)
                    outputs2 = efnet(**inputs2)

                    loss_value = efnet_loss(torch.squeeze(outputs1, dim=1), labels.float())
                    loss_value += efnet_loss(torch.squeeze(outputs2, dim=1), 1.0 - labels)
                    
                    loss.append(loss_value.item())
                    if (i+1)*len(batch['img']) > 20000:
                        # print('val break')
                        break
                    
            # Compute mean loss or handle empty loss list
            if loss:
                epoch_loss = np.mean(loss)
            else:
                logger.info("Loss list is empty.")
                epoch_loss = 0

            schedulerE.step(epoch_loss)

            logger.info("Epoch:%d   Valloss: %.8f"%(epoch, epoch_loss))

            if device == 0 and (np.mean(loss)<best_loss3):
                best_loss3 = np.mean(loss)
                torch.save({'epoch':epoch,
                           'model_state_dict': efnet.module.state_dict(),
                           'optimizer_state_dict':optimizerE.state_dict()
                           },"../Data/risaac6/weights/"+dataset+"/netE_"+dataset+".pth.tar")

        if evaluation_flag:
            efnet_evaluate(efnet, netG, test_img_caption_dataloader, device)
        