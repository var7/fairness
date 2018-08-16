import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import argparse

plt.switch_backend('agg')
# In[3]:
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}

matplotlib.rc('font', **font)

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from torchvision import transforms, utils, models, datasets
from torch.utils.data import Dataset, DataLoader


# In[4]:


from synthetic_utils import *


# In[5]:


from trainer_dataloader import *
from networks import *
from losses import *


# In[6]:


input_size = 96
batch_size = 64
num_workers = 4
num_epochs = 500


cuda = False
pin_memory = False
if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    cuda = True
    cudnn.benchmark = True
    pin_memory = True
else:
    device = torch.device("cpu")

print('Device set: {}'.format(device))


# In[8]:


data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]),
        'val': transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
}


# In[9]:

parser = argparse.ArgumentParser()

parser.add_argument("-j", "--job-number", dest="job_number",
                    help="job number to store weights")

args = parser.parse_args()

DATA_PATH = '/home/var/synthetic_data/dependent_gen/'
TRAIN_PATH = os.path.join(DATA_PATH, 'train')
VAL_PATH = os.path.join(DATA_PATH, 'valid')
TEST_PATH = os.path.join(DATA_PATH, 'test')

WEIGHTS_PATH = './bce_{}/weights'.format(args.job_number)
if not os.path.exists(WEIGHTS_PATH):
    os.makedirs(WEIGHTS_PATH)
PLT_PATH = './bce_{}/plots/'.format(args.job_number)
if not os.path.exists(PLT_PATH):
    os.makedirs(PLT_PATH)

# In[10]:


train_df = datasets.ImageFolder(root=TRAIN_PATH, transform=data_transforms['train'])
val_df = datasets.ImageFolder(root=VAL_PATH, transform=data_transforms['val'])
test_df = datasets.ImageFolder(root=TEST_PATH, transform=data_transforms['val'])


# In[11]:


train_loader = DataLoader(train_df, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
val_loader = DataLoader(val_df, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
test_loader = DataLoader(test_df, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)


# ## LAFTR Training


from synthetic_dataloader import *
shapegender_train = ShapeGenderDataset(train_df)
shapegender_valid = ShapeGenderDataset(val_df)


# In[14]:


laftrtrain_loader = DataLoader(shapegender_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
laftrval_loader = DataLoader(shapegender_valid, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)


# In[15]:


laftr_encoder = LeNet()
laftr_adversary = ClassNet()
laftr_classifier = ClassNet()


# In[16]:


# laftr_adv_criterion = AdvDemographicParityLoss()
laftr_adv_criterion = nn.BCELoss()
laftr_cls_criterion = nn.BCELoss()


# In[17]:


laftr_opt_adv = optim.Adam(laftr_adversary.parameters(), lr=0.001, betas=(0.9, 0.999))
# laftr_opt_cls = optim.Adam(laftr_classifier.parameters(), lr=0.0001, betas=(0.9, 0.999))
laftr_opt_enc = optim.Adam(laftr_encoder.parameters(), lr=0.0001, betas=(0.9, 0.999))

# laftr_opt_adv = optim.SGD(laftr_adversary.parameters(), lr=0.001, momentum=0.9)
laftr_opt_cls = optim.SGD(laftr_classifier.parameters(), lr=0.0001, momentum=0.9)
# laftr_opt_enc = optim.SGD(laftr_encoder.parameters(), lr=0.0001, momentum=0.9)

laftr_scheduler_adv = lr_scheduler.StepLR(optimizer=laftr_opt_adv, gamma=0.99, step_size=1)
laftr_scheduler_cls = lr_scheduler.StepLR(optimizer=laftr_opt_cls, gamma=0.99, step_size=1)
laftr_scheduler_enc = lr_scheduler.StepLR(optimizer=laftr_opt_enc, gamma=0.99, step_size=1)


# In[18]:


clsTrain_losses = []
clsTrain_accs = []
# trainCombined_losses = []
clsTrainCombined_losses = []
advTrain_losses = []
advTrain_accs = []
advTrainCombined_losses = []

combinedVal_losses = []
clsVal_losses = []
clsVal_accs = []
advVal_losses = []
advVal_accs = []

best_acc = 0.7
epoch_time = AverageMeter()


# In[ ]:


ep_end = time.time()
for epoch in range(0, num_epochs):
#         print('-'*80)
        print('Epoch: {}/{}'.format(epoch, num_epochs))

        laftr_scheduler_adv.step()
        laftr_scheduler_cls.step()
        laftr_scheduler_enc.step()
        
        cls_loss, cls_en_acc, adv_loss, adv_acc, cls_en_combinedLoss, adv_combinedLoss = alfr_train_bce(laftrtrain_loader,
                                                        laftr_encoder, laftr_classifier, laftr_adversary, laftr_opt_enc,
                                                        laftr_opt_cls, laftr_opt_adv, 
                                                        laftr_cls_criterion, laftr_adv_criterion, device, 0.65, 0.55)
        
        clsTrain_losses.append(cls_loss)
        clsTrain_accs.append(cls_en_acc)
        clsTrainCombined_losses.append(cls_en_combinedLoss)
        advTrain_losses.append(adv_loss)
        advTrain_accs.append(adv_acc)
#         trainCombined_losses.append(combined_loss)
        advTrainCombined_losses.append(adv_combinedLoss)
        
        print('\nClassifier accuracy: {}\t Adversary Accuracy: {}'.format(cls_en_acc, adv_acc))
        # validate
        print('-'*10)
        
        combinedVal_loss, clsVal_loss, clsVal_acc, advVal_loss, advVal_acc = laftr_validate_dp(laftrval_loader,
                                                        laftr_encoder, laftr_classifier, laftr_adversary, 
                                                        laftr_cls_criterion, laftr_adv_criterion, device)
        
        combinedVal_losses.append(combinedVal_loss)
        clsVal_losses.append(clsVal_loss)
        clsVal_accs.append(clsVal_acc)
        advVal_losses.append(advVal_loss)
        advVal_accs.append(advVal_acc)
        
        print('%'*20)
        print('Classifier validation acc: {:.4f} \t Adv validation acc: {:.4f}'.format(clsVal_acc, advVal_acc))
        
        if clsVal_acc > best_acc and advVal_acc < 0.8:
            print('SAVING WEIGHTS')
            best_acc = clsVal_acc
            torch.save(laftr_encoder, os.path.join(WEIGHTS_PATH, 'encoder_{}_{}.pth'.format(epoch, clsVal_acc)))
            torch.save(laftr_classifier, os.path.join(WEIGHTS_PATH, 'cls_{}_{}.pth'.format(epoch, clsVal_acc)))
            torch.save(laftr_adversary, os.path.join(WEIGHTS_PATH, 'adv_{}_{}.pth'.format(epoch, advVal_acc)))

        print('-' * 20)
        epoch_time.update(time.time() - ep_end)
        ep_end = time.time()
        print('Epoch {}/{}\t'
              'Time {epoch_time.val:.3f} sec ({epoch_time.avg:.3f} sec)'.format(epoch, num_epochs, epoch_time=epoch_time))
        print('-'*20)


# In[ ]:

pkl_path = os.path.join(PLT_PATH, 'metrics.pkl')
with open(pkl_path, 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([advTrainCombined_losses, clsTrainCombined_losses, combinedVal_losses, clsTrain_losses,
                 clsVal_losses, clsTrain_accs, clsVal_accs, advTrain_losses, advVal_losses, advTrain_accs, advVal_accs], f)
    
# plt.figure(figsize=(20,20))
# plt.title('Combined Val Loss')
# # plt.plot(trainCombined_losses, label='Train')
# plt.plot(combinedVal_losses, label='Validation')
# plt.legend()
# plt.tight_layout()
# plt.savefig(os.path.join(PLT_PATH, '1.pdf'))

plt.figure(figsize=(20,20))
plt.title('Cls-Enc Loss')
plt.plot(clsTrain_losses, label='Train')
plt.plot(clsVal_losses, label='Validation')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLT_PATH, 'cls_enc_loss.pdf'))

plt.figure(figsize=(20,20))
plt.title('Cls-Enc Accuracy')
plt.plot(clsTrain_accs, label='Train')
plt.plot(clsVal_accs, label='Validation')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLT_PATH, 'cls_enc_acc.pdf'))

plt.figure(figsize=(20,20))
plt.title('Adv Loss')
plt.plot(advTrain_losses, label='Train')
plt.plot(advVal_losses, label='Validation')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLT_PATH, 'adv_loss.pdf'))

plt.figure(figsize=(20,20))
plt.title('Adv Accuracy')
plt.plot(advTrain_accs, label='Train')
plt.plot(advVal_accs, label='Validation')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLT_PATH, 'adv_acc.pdf'))

plt.figure(figsize=(20,20))
plt.title('Step Loss')
plt.plot(advTrainCombined_losses, label='Adversary')
plt.plot(clsTrainCombined_losses, label='Classifier')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLT_PATH, 'combined_loss.pdf'))




# In[ ]:


gender_train = GenderDataset(train_df)
gender_valid = GenderDataset(val_df)


# In[ ]:


advtrain_loader = DataLoader(gender_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
advval_loader = DataLoader(gender_valid, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)


# In[ ]:


adversary = ClassNet()


# In[ ]:


adv_criterion = nn.BCELoss()
opt_adv = optim.Adam(adversary.parameters(), lr=0.001, betas=(0.9, 0.999))
scheduler_adv = lr_scheduler.StepLR(optimizer=opt_adv, gamma=0.99, step_size=1)


# In[ ]:


num_epochs = 10
train_losses = []
train_accs = []
val_losses = []
val_accs = []
epoch_time = AverageMeter()
ep_end = time.time()
for epoch in range(0, num_epochs):
        print('Epoch: {}/{}'.format(epoch, num_epochs))
        scheduler_adv.step()
        # train
        train_loss, train_acc = train_classifier_epoch(advtrain_loader, laftr_encoder,
                                adversary, opt_adv, adv_criterion, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        # validate
        print('-'*10)
        val_loss, val_acc = validate_classifier_epoch(advval_loader, laftr_encoder, adversary,
                                 adv_criterion, device)

        print('Avg validation loss: {} \t Accuracy: {}'.format(val_loss, val_acc))
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        print('-' * 20)
        epoch_time.update(time.time() - ep_end)
        ep_end = time.time()
        print('Epoch {}/{}\t'
              'Time {epoch_time.val:.3f} sec ({epoch_time.avg:.3f} sec)'.format(epoch, num_epochs, epoch_time=epoch_time))
        print('-'*20)


# In[ ]:

plt.figure(figsize=(20,20))
plt.subplot(221)
plt.title('training classification loss')
plt.plot(train_losses)
plt.subplot(222)
plt.title('training accuracy')
plt.plot(train_accs)
plt.subplot(223)
plt.title('validation loss')
plt.plot(val_losses)
plt.subplot(224)
plt.title('validation accuracy')
plt.plot(val_accs)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLT_PATH, 'on_new_loss.pdf'))

pkl_path = os.path.join(PLT_PATH, 'metrics_newadv.pkl')
with open(pkl_path, 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([train_losses, train_accs, val_losses, val_accs], f)