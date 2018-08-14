import torch
import os
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torchvision import transforms, utils, models
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn.functional as F
import time

from synthetic_utils import AverageMeter

print_freq = 20

def freeze_model(model):
    model.eval()
    for params in model.parameters():
        params.requires_grad = False
        
def unfreeze_model(model):
    model.train()
    for params in model.parameters():
        params.requires_grad = True
        
def train_epoch(dataloader, model, opt, criterion, batch_size=64):
    model.train()
    losses = []
    acc = []
    for batch_idx, (imgs, labels) in enumerate(dataloader):
        y_batch = labels['shape_type']

        opt.zero_grad()
        # (1) Forward
        y_hat = model(imgs)
        # (2) Compute diff
        loss = criterion(y_hat, y_batch)
        # (3) Compute gradients
        loss.backward()
        # (4) update weights
        opt.step()

        losses.append(loss.item())

        preds = torch.round(y_hat.data).squeeze(1).cpu().numpy()
        accuracy = sum(preds == y_batch).cpu().numpy()/len(y_batch)

        acc.append(accuracy)
    return losses, acc



def validate_epoch(dataloader, model, criterion, batch_size=64):
    model.eval()
    losses = []
    acc = []
    for batch_idx, (imgs, labels) in enumerate(dataloader):
            
        with torch.no_grad():
            y_batch = labels['shape_type']
            
            y_hat = model(imgs)
            # (2) Compute diff
            loss = criterion(y_hat, y_batch)
            losses.append(loss.item())

            preds = torch.round(y_hat.data).squeeze(1).cpu().numpy()
            accuracy = sum(preds == y_batch).cpu().numpy()/len(y_batch)

        acc.append(accuracy)
    return losses, acc

def train_encoder_classifier_epoch(dataloader, encoder, classifier, encoder_opt, classifier_opt, criterion, device, batch_size=64):
    encoder.train()
    classifier.train()
    
    losses = AverageMeter()
    acc = AverageMeter()
    
    batch_time = AverageMeter()
    end = time.time()
    batches = len(dataloader)
    
    for batch_idx, (imgs, labels) in enumerate(dataloader):
        y_batch = labels.to(device).float()
        imgs = imgs.to(device).float()
        
        classifier_opt.zero_grad()
        encoder_opt.zero_grad()

        z = encoder(imgs)
        # (1) Forward

        y_hat = classifier(z)
        # (2) Compute diff
        loss = criterion(y_hat, y_batch)
        # (3) Compute gradients
        loss.backward()
        # (4) update weights
        classifier_opt.step()
        encoder_opt.step()

        batch_time.update(time.time() - end)
        end = time.time()

        # (3) Compute gradients
        losses.update(loss.item(), imgs.size(0))

        preds = torch.round(y_hat.data).squeeze(1).cpu().numpy()
        accuracy = sum(preds == y_batch).cpu().numpy()/len(y_batch)
        # print(y_batch)
        # print(preds)
        # print(accuracy)

        acc.update(accuracy, imgs.size(0))
        
        if batch_idx % print_freq == 0:
            print('Batch: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Accuracy {acc.val:.4f} ({acc.avg:.4f})'.format(batch_idx, batches, batch_time=batch_time,
                      loss=losses, acc=acc))
    return losses.avg, acc.avg

def validate_encoder_classifier_epoch(dataloader, encoder, classifier, criterion, device, batch_size=64):
    encoder.eval()
    classifier.eval()
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    batches = len(dataloader)
    with torch.no_grad():
        end = time.time()
        for batch_idx, (imgs, labels) in enumerate(dataloader):
            y_batch = labels.to(device).float()
            imgs = imgs.to(device).float()
            
            z = encoder(imgs)
            # (1) Forward

            y_hat = classifier(z)

            # (2) Compute diff
            loss = criterion(y_hat, y_batch)

            batch_time.update(time.time() - end)
            end = time.time()

            # (3) Compute gradients
            losses.update(loss.item(), imgs.size(0))

            preds = torch.round(y_hat.data).squeeze(1).cpu().numpy()
            accuracy = sum(preds == y_batch).cpu().numpy()/len(y_batch)

            acc.update(accuracy, imgs.size(0))

            if batch_idx % print_freq == 0:
                print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Accuracy {acc.val:.4f} ({acc.avg:.4f})'.format(
                      batch_idx, batches, batch_time=batch_time,
                      loss=losses, acc=acc))

    return losses.avg, acc.avg




def laftr_epoch(dataloader, encoder, classifier, adversary, opt_en, opt_cls, opt_adv, cls_criterion, adv_criterion, device, batch_size=64):
    '''
    one training laftr epoch
    '''
    cls_losses = AverageMeter()
    adv_losses = AverageMeter()
    cls_en_combinedLosses = AverageMeter()
    cls_en_accs = AverageMeter()
    adv_combinedLosses = AverageMeter()
    adv_accs = AverageMeter()
    
    batch_time = AverageMeter()
    end = time.time()
    batches = len(dataloader)
    
    for batch_idx, (imgs, shape, color) in enumerate(dataloader):
        y_batch = shape.to(device).float()
        a_batch = color.to(device).float()
        imgs = imgs.to(device).float()
        
        # fix adversary take gradient step with classifier and encoder
        unfreeze_model(encoder)
        unfreeze_model(classifier)
        z = encoder(imgs)
        y_hat = classifier(z)

        freeze_model(adversary)
        a_fixed = adversary(z)

        opt_cls.zero_grad()
        opt_en.zero_grad()
        
        cls_loss = cls_criterion(y_hat, y_batch)
        adv_loss_fixed = adv_criterion(a_fixed, a_batch, y_batch)
        cls_en_combinedLoss = cls_loss + adv_loss_fixed
        cls_en_combinedLoss.backward()
        opt_cls.step()
        opt_en.step()
        
        cls_losses.update(cls_loss.item(), imgs.shape[0])
        
        
        # fix encoder and classifier and take gradient step with adversary
        freeze_model(encoder)
        freeze_model(classifier)
        z_fixed = encoder(imgs)
        y_hat_fixed = classifier(z_fixed)
        
        unfreeze_model(adversary)
        a_hat = adversary(z_fixed)
        
        opt_adv.zero_grad()
        
        cls_loss_fixed = cls_criterion(y_hat_fixed, y_batch)
        adv_loss = adv_criterion(a_hat, a_batch, y_batch)
        
        adv_combinedLoss = -(cls_loss_fixed + adv_loss)
        adv_combinedLoss.backward()
        
        opt_adv.step()
        
        adv_losses.update(adv_loss.item(), imgs.shape[0])
        
        cls_en_combinedLosses.update(cls_en_combinedLoss.item(), imgs.shape[0])
        adv_combinedLosses.update(adv_combinedLoss.item(), imgs.shape[0])
        
        cls_preds = torch.round(y_hat.data).squeeze(1).cpu().numpy()
        cls_acc = sum(cls_preds == y_batch).cpu().numpy()/len(y_batch)
        cls_en_accs.update(cls_acc, imgs.shape[0])
        
        adv_preds = torch.round(a_hat.data).squeeze(1).cpu().numpy()
        adv_acc = sum(adv_preds == a_batch).cpu().numpy()/len(a_batch)
        adv_accs.update(adv_acc, imgs.shape[0])
        
        batch_time.update(time.time() - end)
        end = time.time()
        
        
        if batch_idx % print_freq == 0:
                print('Batch: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\n'
                      'Classifier loss {cls_loss.val:.4f} ({cls_loss.avg:.4f})\t'
                      'Adversary loss {adv_loss.val:.4f} ({adv_loss.avg:.4f})\n'
                      'Combined Loss during classifier step {cls_combloss.val:.4f} ({cls_combloss.avg:.4f})\t'
                      'Combined Loss during adversary step {adv_combloss.val:.4f} ({adv_combloss.avg:.4f})\n'
                      'Classifier Accuracy {cls_acc.val:.4f} ({cls_acc.avg:.4f})\t'
                      'Adversary Accuracy {adv_acc.val:.4f} ({adv_acc.avg:.4f})'.format(batch_idx, batches, batch_time=batch_time,
                                                                            cls_loss=cls_losses, adv_loss=adv_losses,
                                                                            cls_combloss=cls_en_combinedLosses, 
                                                                            adv_combloss=adv_combinedLosses,
                                                                            cls_acc=cls_en_accs, adv_acc=adv_accs))
                
    return cls_losses.avg, cls_en_combinedLosses.avg, cls_en_accs.avg, adv_losses.avg, adv_combinedLosses.avg, adv_accs.avg

def laftr_validate(dataloader, encoder, classifier, adversary, loss_cls, loss_adv, device, batch_size=64):
    combined_losses = AverageMeter()
    cls_en_losses = AverageMeter()
    cls_en_accs = AverageMeter()
    adv_losses = AverageMeter()
    adv_accs = AverageMeter()
    
    encoder.eval()
    classifier.eval()
    adversary.eval()
    
    batch_time = AverageMeter()
    batches = len(dataloader)
    
    
    with torch.no_grad():
        end = time.time()
        for batch_idx, (imgs, shape, color) in enumerate(dataloader):
            y_cls_batch = shape.to(device).float()
            y_adv_batch = color.to(device).float()
            imgs = imgs.to(device).float()
            
            # fix adversary take gradient step with classifier and encoder
            z = encoder(imgs)
            y_hat = classifier(z)

            a_hat = adversary(z)
        
            cls_en_loss = loss_cls(y_hat, y_cls_batch)
            adv_loss = loss_adv(a_hat, y_adv_batch, y_cls_batch)
            combined_loss = cls_en_loss + adv_loss
        
            cls_en_losses.update(cls_en_loss.item(), imgs.shape[0])
            adv_losses.update(adv_loss.item(), imgs.shape[0])
            combined_losses.update(combined_loss, imgs.shape[0])

            cls_preds = torch.round(y_hat.data).squeeze(1).cpu().numpy()
            cls_acc = sum(cls_preds == y_cls_batch).cpu().numpy()/len(y_cls_batch)
            cls_en_accs.update(cls_acc, imgs.shape[0])

            adv_preds = torch.round(a_hat.data).squeeze(1).cpu().numpy()
            adv_acc = sum(adv_preds == y_adv_batch).cpu().numpy()/len(y_adv_batch)
            adv_accs.update(adv_acc, imgs.shape[0])
            
            batch_time.update(time.time() - end)
            end = time.time()
            
            if batch_idx % print_freq == 0:
                print('Test batch: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\n'
                      'Classifier loss {cls_loss.val:.4f} ({cls_loss.avg:.4f})\t'
                      'Adversary loss {adv_loss.val:.4f} ({adv_loss.avg:.4f})\n'
                      'Combined Loss {combloss.val:.4f} ({combloss.avg:.4f})\t'
                      'Classifier Accuracy {cls_acc.val:.4f} ({cls_acc.avg:.4f})\t'
                      'Adversary Accuracy {adv_acc.val:.4f} ({adv_acc.avg:.4f})'.format(batch_idx, batches, batch_time=batch_time,
                                                                            cls_loss=cls_en_losses, adv_loss=adv_losses,
                                                                            combloss=combined_losses, 
                                                                            cls_acc=cls_en_accs, adv_acc=adv_accs))
                
    return combined_losses.avg, cls_en_losses.avg, cls_en_accs.avg, adv_losses.avg, adv_accs.avg

def laftr_epoch_dp(dataloader, encoder, classifier, adversary, opt_en, opt_cls, opt_adv, cls_criterion, adv_criterion, device, batch_size=64):
    '''
    one training laftr epoch
    '''
    cls_losses = AverageMeter()
    adv_losses = AverageMeter()
    cls_en_combinedLosses = AverageMeter()
    cls_en_accs = AverageMeter()
    adv_combinedLosses = AverageMeter()
    adv_accs = AverageMeter()
    
    batch_time = AverageMeter()
    end = time.time()
    batches = len(dataloader)
    
    for batch_idx, (imgs, shape, color) in enumerate(dataloader):
        y_cls_batch = shape.to(device).float()
        y_adv_batch = color.to(device).float()
        imgs = imgs.to(device).float()
        
        # fix adversary take gradient step with classifier and encoder
        unfreeze_model(encoder)
        unfreeze_model(classifier)
        z = encoder(imgs)
        y_hat = classifier(z)
        
        freeze_model(adversary)
        a_fixed = adversary(z)

        opt_cls.zero_grad()
        opt_en.zero_grad()
        
        cls_loss = cls_criterion(y_hat, y_cls_batch)
        adv_loss_fixed = adv_criterion(a_fixed, y_adv_batch)
        cls_en_combinedLoss = cls_loss + adv_loss_fixed
        cls_en_combinedLoss.backward()
        opt_cls.step()
        opt_en.step()
        
        cls_losses.update(cls_loss.item(), imgs.shape[0])
        
        
        # fix encoder and classifier and take gradient step with adversary
        freeze_model(encoder)
        freeze_model(classifier)
        z_fixed = encoder(imgs)
        y_hat_fixed = classifier(z_fixed)
        
        unfreeze_model(adversary)
        a_hat = adversary(z_fixed)
        
        opt_adv.zero_grad()
        
        cls_loss_fixed = cls_criterion(y_hat_fixed, y_cls_batch)
        adv_loss = adv_criterion(a_hat, y_adv_batch)
        
        adv_combinedLoss = -(cls_loss_fixed + adv_loss)
        adv_combinedLoss.backward()
        
        opt_adv.step()
        
        adv_losses.update(adv_loss.item(), imgs.shape[0])
        
        cls_en_combinedLosses.update(cls_en_combinedLoss.item(), imgs.shape[0])
        adv_combinedLosses.update(adv_combinedLoss.item(), imgs.shape[0])
        
        cls_preds = torch.round(y_hat.data).squeeze(1).cpu().numpy()
        cls_acc = sum(cls_preds == y_cls_batch).cpu().numpy()/len(y_cls_batch)
        cls_en_accs.update(cls_acc, imgs.shape[0])
        
        adv_preds = torch.round(a_hat.data).squeeze(1).cpu().numpy()
        adv_acc = sum(adv_preds == y_adv_batch).cpu().numpy()/len(y_adv_batch)
        adv_accs.update(adv_acc, imgs.shape[0])
        
        batch_time.update(time.time() - end)
        end = time.time()
        
        
        if batch_idx % print_freq == 0:
                print('Batch: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\n'
                      'Classifier loss {cls_loss.val:.4f} ({cls_loss.avg:.4f})\t'
                      'Adversary loss {adv_loss.val:.4f} ({adv_loss.avg:.4f})\n'
                      'Combined Loss during classifier step {cls_combloss.val:.4f} ({cls_combloss.avg:.4f})\t'
                      'Combined Loss during adversary step {adv_combloss.val:.4f} ({adv_combloss.avg:.4f})\n'
                      'Classifier Accuracy {cls_acc.val:.4f} ({cls_acc.avg:.4f})\t'
                      'Adversary Accuracy {adv_acc.val:.4f} ({adv_acc.avg:.4f})'.format(batch_idx, batches, batch_time=batch_time,
                                                                            cls_loss=cls_losses, adv_loss=adv_losses,
                                                                            cls_combloss=cls_en_combinedLosses, 
                                                                            adv_combloss=adv_combinedLosses,
                                                                            cls_acc=cls_en_accs, adv_acc=adv_accs))
                
    return cls_losses.avg, cls_en_combinedLosses.avg, cls_en_accs.avg, adv_losses.avg, adv_combinedLosses.avg, adv_accs.avg

def laftr_validate_dp(dataloader, encoder, classifier, adversary, loss_cls, loss_adv, device, batch_size=64):
    combined_losses = AverageMeter()
    cls_en_losses = AverageMeter()
    cls_en_accs = AverageMeter()
    adv_losses = AverageMeter()
    adv_accs = AverageMeter()
    
    encoder.eval()
    classifier.eval()
    adversary.eval()
    
    batch_time = AverageMeter()
    
    batches = len(dataloader)
    
    
    with torch.no_grad():
        end = time.time()
        for batch_idx, (imgs, shape, color) in enumerate(dataloader):
            y_cls_batch = shape.to(device).float()
            y_adv_batch = color.to(device).float()
            imgs = imgs.to(device).float()
            
            # fix adversary take gradient step with classifier and encoder
            z = encoder(imgs)
            y_hat = classifier(z)

            a_hat = adversary(z)
        
            cls_en_loss = loss_cls(y_hat, y_cls_batch)
            adv_loss = loss_adv(a_hat, y_adv_batch)
            combined_loss = cls_en_loss + adv_loss
        
            cls_en_losses.update(cls_en_loss.item(), imgs.shape[0])
            adv_losses.update(adv_loss.item(), imgs.shape[0])
            combined_losses.update(combined_loss, imgs.shape[0])

            cls_preds = torch.round(y_hat.data).squeeze(1).cpu().numpy()
            cls_acc = sum(cls_preds == y_cls_batch).cpu().numpy()/len(y_cls_batch)
            cls_en_accs.update(cls_acc, imgs.shape[0])

            adv_preds = torch.round(a_hat.data).squeeze(1).cpu().numpy()
            adv_acc = sum(adv_preds == y_adv_batch).cpu().numpy()/len(y_adv_batch)
            adv_accs.update(adv_acc, imgs.shape[0])
            
            batch_time.update(time.time() - end)
            end = time.time()
            
            if batch_idx % print_freq == 0:
                print('Test batch: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\n'
                      'Classifier loss {cls_loss.val:.4f} ({cls_loss.avg:.4f})\t'
                      'Adversary loss {adv_loss.val:.4f} ({adv_loss.avg:.4f})\n'
                      'Combined Loss {combloss.val:.4f} ({combloss.avg:.4f})\t'
                      'Classifier Accuracy {cls_acc.val:.4f} ({cls_acc.avg:.4f})\t'
                      'Adversary Accuracy {adv_acc.val:.4f} ({adv_acc.avg:.4f})'.format(batch_idx, batches, batch_time=batch_time,
                                                                            cls_loss=cls_en_losses, adv_loss=adv_losses,
                                                                            combloss=combined_losses, 
                                                                            cls_acc=cls_en_accs, adv_acc=adv_accs))
                
    return combined_losses.avg, cls_en_losses.avg, cls_en_accs.avg, adv_losses.avg, adv_accs.avg


def train_classifier_epoch(dataloader, encoder, classifier, classifier_opt, criterion, device, batch_size=64):
    encoder.eval()
    classifier.train()
    
    losses = AverageMeter()
    acc = AverageMeter()
    
    batch_time = AverageMeter()
    end = time.time()
    batches = len(dataloader)
    
    for batch_idx, (imgs, labels) in enumerate(dataloader):
        y_batch = labels.to(device).float()
        imgs = imgs.to(device).float()
        
        classifier_opt.zero_grad()
        z = encoder(imgs)
        # (1) Forward
        y_hat = classifier(z)
        # (2) Compute diff
        loss = criterion(y_hat, y_batch)
        # (3) Compute gradients
        loss.backward()
        # (4) update weights
        classifier_opt.step()

        batch_time.update(time.time() - end)
        end = time.time()

        # (3) Compute gradients
        losses.update(loss.item(), imgs.size(0))

        preds = torch.round(y_hat.data).squeeze(1).cpu().numpy()
        accuracy = sum(preds == y_batch).cpu().numpy()/len(y_batch)
        # print(y_batch)
        # print(preds)
        # print(accuracy)

        acc.update(accuracy, imgs.size(0))
        
        if batch_idx % print_freq == 0:
            print('Batch: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Accuracy {acc.val:.4f} ({acc.avg:.4f})'.format(batch_idx, batches, batch_time=batch_time,
                      loss=losses, acc=acc))
    return losses.avg, acc.avg

def validate_classifier_epoch(dataloader, encoder, classifier, criterion, device, batch_size=64):
    encoder.eval()
    classifier.eval()
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    batches = len(dataloader)
    with torch.no_grad():
        end = time.time()
        for batch_idx, (imgs, labels) in enumerate(dataloader):
            y_batch = labels.to(device).float()
            imgs = imgs.to(device).float()
            
            z = encoder(imgs)
            # (1) Forward

            y_hat = classifier(z)

            # (2) Compute diff
            loss = criterion(y_hat, y_batch)

            batch_time.update(time.time() - end)
            end = time.time()

            # (3) Compute gradients
            losses.update(loss.item(), imgs.size(0))

            preds = torch.round(y_hat.data).squeeze(1).cpu().numpy()
            accuracy = sum(preds == y_batch).cpu().numpy()/len(y_batch)

            acc.update(accuracy, imgs.size(0))

            if batch_idx % print_freq == 0:
                print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Accuracy {acc.val:.4f} ({acc.avg:.4f})'.format(
                      batch_idx, batches, batch_time=batch_time,
                      loss=losses, acc=acc))

    return losses.avg, acc.avg

def alfr_train(dataloader, encoder, classifier, adversary, opt_en, opt_cls, opt_adv, cls_criterion, adv_criterion, device, batch_size=64):
    cls_losses = AverageMeter()
    adv_losses = AverageMeter()
    cls_en_combinedLosses = AverageMeter()
    cls_en_accs = AverageMeter()
    adv_combinedLosses = AverageMeter()
    adv_accs = AverageMeter()
    # combined_loss = AverageMeter()
    
    batch_time = AverageMeter()
    end = time.time()
    batches = len(dataloader)
    
    updateEncoder = False
    for batch_idx, (imgs, shape, color) in enumerate(dataloader):
        y_cls_batch = shape.to(device).float()
        y_adv_batch = color.to(device).float()
        imgs = imgs.to(device).float()
        
        # zero out accumulated gradients
        opt_cls.zero_grad()
        opt_en.zero_grad()
        opt_adv.zero_grad()

        # fix adversary take gradient step with classifier and encoder
        z = encoder(imgs)
        y_hat = classifier(z)
        a_hat = adversary(z)
        
        cls_loss = cls_criterion(y_hat.squeeze(), y_cls_batch)
        adv_loss = adv_criterion(a_hat.squeeze(), y_adv_batch)
        
        cls_preds = torch.round(y_hat.data).squeeze(1).cpu().numpy()
        cls_acc = sum(cls_preds == y_cls_batch).cpu().numpy()/len(y_cls_batch)
        cls_en_accs.update(cls_acc, imgs.shape[0])
        
        adv_preds = torch.round(a_hat.data).squeeze(1).cpu().numpy()
        adv_acc = sum(adv_preds == y_adv_batch).cpu().numpy()/len(y_adv_batch)
        adv_accs.update(adv_acc, imgs.shape[0])
        
        if (updateEncoder and adv_acc > 0.6) or (updateEncoder and cls_acc < 0.45):
            print('*', end='')
            # if adv_acc < 0.6: 
            #     # print('Skipping encoder update')
            #     continue
            combinedLoss = cls_loss - adv_loss
            combinedLoss.backward()
            opt_en.step()
            opt_cls.step()
            cls_en_combinedLosses.update(combinedLoss.item(), imgs.shape[0])
            
        elif not updateEncoder and adv_acc < 0.9:
            print('$', end='')
            # if adv_acc > 0.9:
            #     # print('Skipping adv update')
            #     continue
            combinedLoss = adv_loss - cls_loss 
            combinedLoss.backward()
            opt_adv.step()
            adv_combinedLosses.update(combinedLoss.item(), imgs.shape[0])
        
        cls_losses.update(cls_loss.item(), imgs.shape[0])
        adv_losses.update(adv_loss.item(), imgs.shape[0])
        
#         cls_en_combinedLosses.update(cls_en_combinedLoss.item(), imgs.shape[0])
#         adv_combinedLosses.update(adv_combinedLoss.item(), imgs.shape[0])
        
#         cls_preds = torch.round(y_hat.data).squeeze(1).cpu().numpy()
#         cls_acc = sum(cls_preds == y_cls_batch).cpu().numpy()/len(y_cls_batch)
#         cls_en_accs.update(cls_acc, imgs.shape[0])
        
#         adv_preds = torch.round(a_hat.data).squeeze(1).cpu().numpy()
#         adv_acc = sum(adv_preds == y_adv_batch).cpu().numpy()/len(y_adv_batch)
#         adv_accs.update(adv_acc, imgs.shape[0])
        
        batch_time.update(time.time() - end)
        end = time.time()
        
        
        if batch_idx % print_freq == 0:
                print('\nBatch: [{0}/{1}]\t'
                      # 'Step -  Encoder: {2}\n'
                      #'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\n'
                      'Cls step loss:{cls_loss.val:.4f} ({cls_loss.avg:.4f})\t'
                      'Adv step loss:{adv_loss.val:.4f} ({adv_loss.avg:.4f})\n'
                      # 'Combined loss {combined_loss.val:.4f} ({combined_loss.avg:.4f})\n'
                      'Cls Acc:{cls_acc.val:.4f} ({cls_acc.avg:.4f})\t'
                      'Adv Acc:{adv_acc.val:.4f} ({adv_acc.avg:.4f})'.format(batch_idx, batches, #updateEncoder, 
                                                                            #batch_time=batch_time,
                                                                            # cls_loss=cls_losses, adv_loss=adv_losses,
                                                                            cls_loss = cls_en_combinedLosses, adv_loss=adv_combinedLosses,
                                                                            # combined_loss=combined_loss,
                                                                            cls_acc=cls_en_accs, adv_acc=adv_accs))
        updateEncoder = not updateEncoder
        
    return cls_losses.avg, cls_en_accs.avg, adv_losses.avg, adv_accs.avg, 0