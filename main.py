import os

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import torch
import torch.nn as nn
from torch import optim
from torchcontrib.optim import SWA

from dataloader_MMC import generate_dataloader
from model.T2ID import Base_Model
from utils_MMC import adjust_learning_rate, CraateLogger, create_cosine_learing_schdule, set_seed


def criterion(logit, truth):
    return nn.CrossEntropyLoss()(logit, truth)


def metric(logit, truth):
    _, prediction = torch.max(logit.data, 1)
    return torch.sum(prediction == truth)


def train(net, train_dataloader):
    net.train()
    train_acc = 0

    for index, (clinic_image, derm_image, meta_data, label) in enumerate(train_dataloader):
        opt.zero_grad()

        clinic_image = clinic_image.cuda()
        derm_image = derm_image.cuda()
        diagnosis_label = label[1].long().cuda()

        logits, _, confidence_loss = net(clinic_image, derm_image, diagnosis_label, update_memory=True)

        loss = criterion(logits, diagnosis_label) + confidence_loss
        acc = torch.true_divide(metric(logits, diagnosis_label), clinic_image.size(0))

        loss.backward()
        opt.step()

        train_acc += acc.item()

    return train_acc / (index + 1)


def validation(net, val_dataloader):
    net.eval()
    val_acc = 0

    for index, (clinic_image, derm_image, meta_data, label) in enumerate(val_dataloader):
        clinic_image = clinic_image.cuda()
        derm_image = derm_image.cuda()
        diagnosis_label = label[1].long().cuda()

        with torch.no_grad():
            logits, _, _ = net(clinic_image, derm_image, diagnosis_label, update_memory=False)
            acc = torch.true_divide(metric(logits, diagnosis_label), clinic_image.size(0))

        val_acc += acc.item()

    return val_acc / (index + 1)


def run_train(i):
    for epoch in range(epochs):
        swa_lr = cosine_learning_schule[epoch]
        adjust_learning_rate(opt, swa_lr)

        train_acc = train(net, train_dataloader)
        val_acc = validation(net, val_dataloader)

        log.write('Round: {}, epoch: {}, Train ACC: {:.4f}, Valid ACC: {:.4f}\n'.format(
            i, epoch, train_acc, val_acc
        ))
        if epoch > (epochs - swa_epoch):
            opt.update_swa()


if __name__ == '__main__':
    mode = 'multimodal'
    model_name = 'Our-MMC_r1_missing=0.6_complete_experiments_lr_1e-5_seed_42_update_memory_bank'
    shape = (224, 224)
    batch_size = 32
    num_workers = 8
    data_mode = 'Normal'
    deterministic = True
    random_seeds = 42 if data_mode == 'Normal' else 183
    rounds = 1
    lr = 1e-5
    epochs = 250
    swa_epoch = 50

    train_dataloader, val_dataloader = generate_dataloader(shape, batch_size, num_workers, data_mode)

    for i in range(rounds):
        if deterministic:
            set_seed(random_seeds + i)

        log, out_dir = CraateLogger(mode, model_name, i, data_mode)
        net = Base_Model(num_classes=4, p_missing=0.6, memory_size=500).cuda()
        optimizer = optim.Adam(net.parameters(), lr=lr)
        opt = SWA(optimizer)
        cosine_learning_schule = create_cosine_learing_schdule(epochs, lr)
        run_train(i)
