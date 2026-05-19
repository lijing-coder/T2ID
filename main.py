import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import torch
import torch.nn as nn
from torch import optim

from dataloader_MMC import generate_dataloader
from utils_MMC import adjust_learning_rate, CraateLogger, create_cosine_learing_schdule, set_seed
from model.T2ID import T2ID
from dependency_MMC import *

def criterion(logit, truth):
    return nn.CrossEntropyLoss()(logit, truth)

def metric(logit, truth):
    _, prediction = torch.max(logit.data, 1)
    acc = torch.sum(prediction == truth)
    return acc

def train(net, train_dataloader):
    net.train()
    train_loss = 0.0
    train_dia_acc = 0.0

    for index, (clinic_image, derm_image, meta_data, label) in enumerate(train_dataloader):
        opt.zero_grad()

        clinic_image = clinic_image.cuda()
        derm_image = derm_image.cuda()
        diagnosis_label = label[1].long().cuda()

        logit_diagnosis_fusion, confidence_loss = net(
            clinic_image,
            derm_image,
            diagnosis_label,
            update_memory=True
        )

        loss_fusion = criterion(logit_diagnosis_fusion, diagnosis_label)
        loss = loss_fusion + confidence_loss

        dia_acc = torch.true_divide(
            metric(logit_diagnosis_fusion, diagnosis_label),
            clinic_image.size(0)
        )

        loss.backward()
        opt.step()

        train_loss += loss.item()
        train_dia_acc += dia_acc.item()

    train_loss = train_loss / (index + 1)
    train_dia_acc = train_dia_acc / (index + 1)

    return train_loss, train_dia_acc


def validation(net, val_dataloader):
    net.eval()
    val_loss = 0.0
    val_dia_acc = 0.0

    with torch.no_grad():
        for index, (clinic_image, derm_image, meta_data, label) in enumerate(val_dataloader):
            clinic_image = clinic_image.cuda()
            derm_image = derm_image.cuda()
            diagnosis_label = label[1].long().cuda()

            logit_diagnosis_fusion, confidence_loss = net(
                clinic_image,
                derm_image,
                diagnosis_label,
                update_memory=False
            )

            loss = criterion(logit_diagnosis_fusion, diagnosis_label) + confidence_loss
            acc = torch.true_divide(
                metric(logits, diagnosis_label),
                clinic_image.size(0)
            )

            val_loss += loss.item()
            val_dia_acc += acc.item()

    val_loss = val_loss / (index + 1)
    val_dia_acc = val_dia_acc / (index + 1)

    return val_loss, val_dia_acc


def save_best_checkpoint(net, epoch, best_acc, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "best_acc": best_acc,
            "model_state_dict": net.state_dict(),
        },
        save_path
    )


def run_train(model_name, mode, round_id):
    log.write("** start training here! **\n")

    best_acc = 0.0
    best_model_path = os.path.join(out_dir, "checkpoint", f"{model_name}_best_acc.pth")

    for epoch in range(epochs):
        current_lr = cosine_learning_schule[epoch]
        adjust_learning_rate(opt, current_lr)

        train_loss, train_dia_acc = train(net, train_dataloader)
        log.write(
            "Round: {}, epoch: {}, Train Loss: {:.4f}, Train Dia Acc: {:.4f}\n".format(
                round_id, epoch, train_loss, train_dia_acc
            )
        )

        val_loss, val_dia_acc = validation(net, val_dataloader)
        log.write(
            "Round: {}, epoch: {}, Valid Loss: {:.4f}, Valid Dia Acc: {:.4f}\n".format(
                round_id, epoch, val_loss, val_dia_acc
            )
        )

        if val_dia_acc > best_acc:
            best_acc = val_dia_acc
            save_best_checkpoint(net, epoch, best_acc, best_model_path)
            log.write(
                "Saved best model at epoch {} with Acc: {:.4f}\n".format(
                    epoch, best_acc
                )
            )

    log.write("Training finished. Best Acc: {:.4f}\n".format(best_acc))
    log.write("Best checkpoint saved to: {}\n".format(best_model_path))


if __name__ == "__main__":
    mode = "multimodal"
    model_name = "T2ID"
    shape = (224, 224)
    batch_size = 32
    num_workers = 8
    data_mode = "Normal"
    deterministic = True
    random_seeds = 42
    rounds = 1
    lr = 1e-5
    epochs = 250

    train_dataloader, val_dataloader = generate_dataloader(
        shape,
        batch_size,
        num_workers,
        data_mode
    )

    for i in range(rounds):
        if deterministic:
            set_seed(random_seeds + i)

        print(random_seeds + i)

        log, out_dir = CraateLogger(mode, model_name, i, data_mode)
        net = Base_Model(
            num_classes=4,
            p_missing=0.6,
            memory_size=500
        ).cuda()

        opt = optim.Adam(net.parameters(), lr=lr)
        cosine_learning_schule = create_cosine_learing_schdule(epochs, lr)

        run_train(model_name, mode, i)
