import os
import logging
import torch
import numpy as np
import pandas as pd
from numpy import sqrt, argmax
from utils.bert_utils import init_bert_model
from sklearn.metrics import auc, roc_curve
from torchvision.utils import save_image

def get_fpr_tpr_auc(Y_label, Y_preds): 
    fpr, tpr, thresholds = roc_curve(Y_label, Y_preds)
    aucscore = auc(fpr, tpr)
    gmeans = sqrt(tpr * (1-fpr))
    ix = argmax(gmeans)
    logger = logging.getLogger()
    logger.info('Best Threshold=%f, G-Mean=%.3f, FPR=%.3f, TPR=%.3f, AUC=%.3f' % (thresholds[ix], gmeans[ix], fpr[ix], tpr[ix], aucscore))
    return fpr, tpr, aucscore


def cvae_evaluate(embnet, recon_loss, test_dataloader, device, variational_beta, imgSize, channel, cvae_batch_size):
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



def cvad_evaluate(embnet, cls_model, recon_loss, cls_loss, test_dataloader, device, variational_beta, imgSize, channel, cvae_batch_size, epoch, normal_class):
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
    
    data = {
        'Target': Y_label,
        'AnomalyScore1': [s[0] for s in anomaly_score1],
        'AnomalyScore2': [s[0] for s in anomaly_score2],
        'CombinedScore': [s[0] for s in Y_preds]
    }
    df = pd.DataFrame(data)
    csv_filename = f"/home/risaac6/Data/risaac6/anomaly_scores/evaluation_results_epoch_{epoch}_{device}_{normal_class}.csv"
    df.to_csv(csv_filename, index=False)
    
    fpr, tpr, aucscore = get_fpr_tpr_auc(Y_label, Y_preds)
    # cls_writer.add_scalar('evaluation', {'fpr':fpr, 'tpr':tpr, 'auc':aucscore}, epoch)
    return fpr, tpr, aucscore

def efnet_evaluate(efnet, embnet, test_img_caption_dataloader, device, save_dir="/home/risaac6/Data/risaac6/efnet_outputs"):
    
    logger = logging.getLogger()
    logger.info("----------- EFNET evaluating------------")
    
    all_outputs = []
    all_labels = []
    
    with torch.no_grad():
        for (i, batch) in enumerate(test_img_caption_dataloader):
            # parse inputs
            batch['img'] = batch['img'].to(device)
            batch['embeddings'] = batch['embeddings'].to(device)
            batch['caption_label'] = batch['caption_label'].to(device)
            images = batch['img']
            embeddings = batch['embeddings']
            labels = batch['caption_label']
            captions = batch['caption']

            recon_x, mu, logvar, mu2, logvar2 = embnet(images)
            inputs = {'image_emb':mu, 'text_emb':embeddings.squeeze(1)}
            outputs = efnet(**inputs)
            
            if i <= 2 and device==0:
                save_images_and_labels(images, labels, captions, outputs.squeeze(), i, save_dir)
            
            # Accumulate outputs and labels on the device
            all_outputs.append(outputs.squeeze())
            all_labels.append(labels)

    # Convert to CPU only once after the loop
    all_outputs_tensor = torch.cat(all_outputs).cpu()
    all_labels_tensor = torch.cat(all_labels).cpu()

    fpr, tpr, aucscore = get_fpr_tpr_auc(all_labels_tensor.numpy(), all_outputs_tensor.numpy())
    return fpr, tpr, aucscore


def save_images_and_labels(images, labels, captions, outputs, batch_num, save_dir):
    """
    Save images and their labels and predictions to files.

    Args:
    images: Tensor of images.
    labels: Actual labels.
    outputs: Predicted labels.
    batch_num: Current batch number for filename uniqueness.
    save_dir: Directory to save the images and labels.
    """
    # Save images
    for idx, image in enumerate(images):
        save_image(image, os.path.join(save_dir, f'batch_{batch_num}_image_{idx}.png'))

    # Prepare data for CSV
    data = {
        'batch_num': [batch_num] * len(images),
        'image_index': list(range(len(images))),
        'caption':captions,
        'label': labels.cpu().numpy(),
        'prediction': outputs.cpu().numpy(),
    }
    df = pd.DataFrame(data)

    # Save or append CSV file
    csv_path = os.path.join(save_dir, 'labels_predictions.csv')
    if os.path.exists(csv_path):
        df.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        df.to_csv(csv_path, index=False)


