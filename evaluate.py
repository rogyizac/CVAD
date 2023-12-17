import logging
import torch
import numpy as np
from numpy import sqrt, argmax
from sklearn.metrics import auc, roc_curve

def get_fpr_tpr_auc(Y_label, Y_preds): 
    fpr, tpr, thresholds = roc_curve(Y_label, Y_preds)
    aucscore = auc(fpr, tpr)
    gmeans = sqrt(tpr * (1-fpr))
    ix = argmax(gmeans)
    logger = logging.getLogger()
    logger.info('Best Threshold=%f, G-Mean=%.3f, FPR=%.3f, TPR=%.3f, AUC=%.3f' % (thresholds[ix], gmeans[ix], fpr[ix], tpr[ix], aucscore))
    return fpr, tpr, aucscore


# def cvae_evaluate(embnet, recon_loss, test_dataloader, device, variational_beta, imgSize, channel, cvae_batch_size):
#     logger = logging.getLogger()
#     logger.info("----------- CVAE evaluating------------")
#     Targets = []
#     anomaly_score = []

#     with torch.set_grad_enabled(False):   
#         for idx, (images, targets) in enumerate(test_dataloader):
#             images = images.to(device)

#             for i in range(0, images.shape[0]):
#                 recon_x, mu, logvar, mu2, logvar2 = embnet(images[i].unsqueeze(0))
#                 cvae_loss = recon_loss(recon_x, images[i].unsqueeze(0), mu, logvar, mu2, logvar2, variational_beta, imgSize, channel, cvae_batch_size)

#                 if not np.isnan(cvae_loss.item()) and not np.isinf(cvae_loss.item()):
#                     anomaly_score.append(cvae_loss.item())
#                     Targets.append(targets[i].detach().cpu().numpy())
            
#     Y_label = np.array(np.vstack(Targets).squeeze(1),dtype=int).tolist() 
#     Y_preds = []
#     for s in anomaly_score:
#         Y_preds.append((s-np.min(np.array(anomaly_score)))/(np.max(np.array(anomaly_score))-np.min(np.array(anomaly_score))))
#     fpr, tpr, aucscore = get_fpr_tpr_auc(Y_label, Y_preds) 
#     return fpr, tpr, aucscore

def cvae_evaluate(embnet, recon_loss, test_dataloader, device, variational_beta, imgSize, channel, cvae_batch_size, epoch):
    logger = logging.getLogger()
    logger.info("----------- CVAE evaluating------------")
    Targets = torch.Tensor().to(device) # initialize on GPU
    anomaly_score = torch.Tensor().to(device) # initialize on GPU

    with torch.no_grad():  # Recommended to use torch.no_grad instead of torch.set_grad_enabled(False)
        for idx, (images, targets) in enumerate(test_dataloader):
            images, targets = images.to(device), targets.to(device)

            for i in range(images.shape[0]):
                recon_x, mu, logvar, mu2, logvar2 = embnet(images[i].unsqueeze(0))
                cvae_loss = recon_loss(recon_x, images[i].unsqueeze(0), mu, logvar, mu2, logvar2, variational_beta, imgSize, channel, cvae_batch_size)

                if not torch.isnan(cvae_loss) and not torch.isinf(cvae_loss):
                    anomaly_score = torch.cat((anomaly_score, cvae_loss.unsqueeze(0)))  # Accumulate on GPU
                    Targets = torch.cat((Targets, targets[i].unsqueeze(0)))  # Accumulate on GPU

    # Transfer to CPU and convert to numpy for sklearn
    Y_label = Targets.detach().cpu().numpy().astype(int).tolist()
    Y_preds = ((anomaly_score - torch.min(anomaly_score)) / (torch.max(anomaly_score) - torch.min(anomaly_score))).detach().cpu().numpy().tolist()
    fpr, tpr, aucscore = get_fpr_tpr_auc(Y_label, Y_preds)
    # cvae_writer.add_scalar('evaluation', {'fpr':fpr, 'tpr':tpr, 'auc':aucscore}, epoch)
    return fpr, tpr, aucscore



def cvad_evaluate(embnet, cls_model, recon_loss, cls_loss, test_dataloader, device, variational_beta, imgSize, channel, cvae_batch_size, epoch):
    logger = logging.getLogger()
    logger.info("----------- CVAD evaluating------------")
    Targets = []
    anomaly_score1 = []
    anomaly_score2 = []

    with torch.set_grad_enabled(False):    
        for idx, (images, targets)  in enumerate(test_dataloader):

            images = images.to(device)

            for i in range(0, images.shape[0]):
                recon_x, mu, logvar, mu2, logvar2 = embnet(images[i].unsqueeze(0))
                outputs = cls_model(images[i].unsqueeze(0))
                cvae_loss = recon_loss(recon_x, images[i].unsqueeze(0), mu, logvar, mu2, logvar2, variational_beta, imgSize, channel, cvae_batch_size)

                if not np.isnan(cvae_loss.item()+outputs.detach().cpu().numpy()[0][0]) and not np.isinf(cvae_loss.item()+outputs.detach().cpu().numpy()[0][0]):
                    anomaly_score1.append([cvae_loss.item()])
                    anomaly_score2.append([outputs.detach().cpu().numpy()[0][0]])
                    Targets.append(targets[i].detach().cpu().numpy())
            
    Y_label = np.array(np.vstack(Targets).squeeze(1),dtype=int).tolist()
    Y_preds = []
    for s1, s2 in zip(anomaly_score1, anomaly_score2):
        Y_preds.append(0.5*((s1-np.min(np.array(anomaly_score1)))/(np.max(np.array(anomaly_score1))-np.min(np.array(anomaly_score1))) + s2))
    aucscore = None
    fpr, tpr, aucscore = get_fpr_tpr_auc(Y_label, Y_preds)
    # cls_writer.add_scalar('evaluation', {'fpr':fpr, 'tpr':tpr, 'auc':aucscore}, epoch)
    return fpr, tpr, aucscore


# def cvad_evaluate(embnet, cls_model, recon_loss, cls_loss, test_dataloader, device, variational_beta, imgSize, channel, cvae_batch_size):
#     logger = logging.getLogger()
#     logger.info("----------- CVAD evaluating------------")
#     Targets = torch.Tensor().to(device) # initialize on GPU
#     anomaly_score1 = torch.Tensor().to(device) # initialize on GPU
#     anomaly_score2 = torch.Tensor().to(device) # initialize on GPU

#     with torch.no_grad():  # Using torch.no_grad for evaluation
#         for idx, (images, targets) in enumerate(test_dataloader):
#             images, targets = images.to(device), targets.to(device)

#             for i in range(images.shape[0]):
#                 recon_x, mu, logvar, mu2, logvar2 = embnet(images[i].unsqueeze(0))
#                 outputs = cls_model(images[i].unsqueeze(0))
#                 cvae_loss = recon_loss(recon_x, images[i].unsqueeze(0), mu, logvar, mu2, logvar2, variational_beta, imgSize, channel, cvae_batch_size)

#                 if not torch.isnan(cvae_loss + outputs[0][0]) and not torch.isinf(cvae_loss + outputs[0][0]):
#                     anomaly_score1 = torch.cat((anomaly_score1, cvae_loss.unsqueeze(0)))  # Accumulate on GPU
#                     anomaly_score2 = torch.cat((anomaly_score2, outputs[:,0].unsqueeze(0)))  # Accumulate on GPU
#                     Targets = torch.cat((Targets, targets[i].unsqueeze(0)))  # Accumulate on GPU

#     # Normalize scores and prepare for sklearn
#     Y_label = Targets.detach().cpu().numpy().astype(int).tolist()
#     anomaly_score1_min = torch.min(anomaly_score1)
#     anomaly_score1_max = torch.max(anomaly_score1)
#     normalized_anomaly_score1 = (anomaly_score1 - anomaly_score1_min) / (anomaly_score1_max - anomaly_score1_min)
#     Y_preds = (0.5 * (normalized_anomaly_score1 + anomaly_score2)).detach().cpu().numpy().tolist()

#     fpr, tpr, aucscore = get_fpr_tpr_auc(Y_label, Y_preds)
#     return fpr, tpr, aucscore

# def cvad_evaluate(embnet, cls_model, recon_loss, cls_loss, test_dataloader, device, variational_beta, imgSize, channel, cvae_batch_size):
#     logger = logging.getLogger()
#     logger.info("----------- CVAD evaluating------------")

#     Targets = []
#     anomaly_score1 = []
#     anomaly_score2 = []

#     with torch.no_grad():  # Disabling gradient computation
#         for images, targets in test_dataloader:
#             images = images.to(device)

#             recon_x, mu, logvar, mu2, logvar2 = embnet(images)
#             outputs = cls_model(images)
#             cvae_loss = recon_loss(recon_x, images, mu, logvar, mu2, logvar2, variational_beta, imgSize, channel, cvae_batch_size)

#             valid_indices = ~(torch.isnan(cvae_loss) | torch.isinf(cvae_loss))
#             valid_targets = targets[valid_indices]

#             anomaly_score1.append(cvae_loss[valid_indices].unsqueeze(1))  # Keep as tensor
#             anomaly_score2.append(outputs[valid_indices].unsqueeze(1))  # Keep as tensor
#             Targets.append(valid_targets.unsqueeze(1))

#     # Concatenating and transferring to CPU in one go
#     anomaly_score1 = torch.cat(anomaly_score1, dim=0).cpu()
#     anomaly_score2 = torch.cat(anomaly_score2, dim=0).cpu()
#     Targets = torch.cat(Targets, dim=0).cpu().numpy().squeeze(1)

#     # Normalizing scores
#     min_score1 = torch.min(anomaly_score1)
#     max_score1 = torch.max(anomaly_score1)
#     normalized_scores = 0.5 * ((anomaly_score1 - min_score1) / (max_score1 - min_score1) + anomaly_score2)

#     Y_label = np.array(Targets, dtype=int)
#     Y_preds = normalized_scores.numpy().tolist()

#     fpr, tpr, aucscore = get_fpr_tpr_auc(Y_label, Y_preds)
#     return fpr, tpr, aucscore