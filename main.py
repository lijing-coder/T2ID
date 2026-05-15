import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
from dataloader_MMC import generate_dataloader
from utils_MMC import Logger, adjust_learning_rate, CraateLogger,create_cosine_learing_schdule,encode_test_label,set_seed
from model.T2ID import Base_Model
from dependency_MMC import *
from torch import optim
from torchcontrib.optim import SWA
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import sklearn.metrics as metrics
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize

#tsne_dir = '/19962387/lijing/mmif-scence-class/image_fusion_moe/MLSDR/constractive_model/'
# class_names = ['NV', 'BCC', 'MEL','MISC','SK']
class_names = ['wetAMD', 'dryAMD', 'PCV', 'Normal']

#绘制confidence和ACC之间的关系
def plot_reliability_diagram_with_boxplot(confidence, pred, target, num_bins=10,
                                          save_path_reliability=None, save_path_boxplot=None):
    # 设置样式
    sns.set(style="whitegrid")
    #plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12

    # 转 tensor
    confidence = torch.tensor(confidence)
    pred = torch.tensor(pred)
    target = torch.tensor(target)
    correct = (pred == target).float()

    # 分 bin
    bins = torch.linspace(0, 1, steps=num_bins + 1)
    bin_centers = ((bins[:-1] + bins[1:]) / 2).numpy()

    acc_per_bin = []
    conf_per_bin = []
    gaps_per_bin = [[] for _ in range(num_bins)]
    ece = 0.0
    N = len(confidence)

    for i in range(num_bins):
        bin_lower = bins[i].item()
        bin_upper = bins[i + 1].item()
        mask = (confidence >= bin_lower) & (confidence < bin_upper)
        bin_size = mask.sum().item()
        if bin_size > 0:
            acc = correct[mask].mean().item()
            conf = confidence[mask].mean().item()
            acc_per_bin.append(acc)
            conf_per_bin.append(conf)
            gap = torch.abs(correct[mask] - confidence[mask]).tolist()
            gaps_per_bin[i].extend(gap)
            ece += (bin_size / N) * abs(acc - conf)
        else:
            acc_per_bin.append(0.0)
            conf_per_bin.append(0.0)
            gaps_per_bin[i] = []

    # ----------------------------
    # 图一：Reliability Diagram
    # ----------------------------
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    ax1.bar(bin_centers, acc_per_bin, width=0.08, color="#70A1D7", edgecolor='black', label='Accuracy', alpha=0.9)
    ax1.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=2, label='Perfect Calibration')
    for i in range(num_bins):
        ax1.plot([conf_per_bin[i], conf_per_bin[i]], [acc_per_bin[i], conf_per_bin[i]], 'r--', linewidth=1)

    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1.05)
    ax1.set_xlabel('Confidence')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Reliability Diagram (Fusion Confidence)\nECE = {:.3f}'.format(ece))
    ax1.legend(loc='upper left')
    plt.tight_layout()

    if save_path_reliability:
        os.makedirs(os.path.dirname(save_path_reliability), exist_ok=True)
        fig1.savefig(save_path_reliability, dpi=300, bbox_inches='tight')
        plt.close(fig1)
    else:
        plt.show()

    # ----------------------------
    # 图二：Calibration Gap Boxplot
    # ----------------------------
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    sns.boxplot(data=gaps_per_bin, ax=ax2, color="#F5B7B1", width=0.6)
    ax2.set_xticks(np.arange(num_bins))
    ax2.set_xticklabels([f"{c:.2f}" for c in bin_centers])
    ax2.set_xlabel('Confidence Bin Center')
    ax2.set_ylabel('|Accuracy - Confidence| (Calibration Gap)')
    ax2.set_title('Calibration Gap Distribution per Bin')
    ax2.set_ylim(0, 1.0)
    plt.tight_layout()

    if save_path_boxplot:
        os.makedirs(os.path.dirname(save_path_boxplot), exist_ok=True)
        fig2.savefig(save_path_boxplot, dpi=300, bbox_inches='tight')
        plt.close(fig2)
    else:
        plt.show()

def plot_kde_three_and_save(df, col1, col2, col3,
                            label1='Feature 1', label2='Feature 2', label3='Feature 3',
                            color1='r', color2='b', color3='g',
                            save_path='kde_plot.png',
                            title='KDE Plot', dpi=300):
    """
    绘制三个特征的KDE图，并保存为图片

    参数：
        df : DataFrame
            包含要绘图数据的DataFrame。
        col1, col2, col3 : str
            要绘制的三个特征列名。
        label1, label2, label3 : str
            每个特征对应的图例标签。
        color1, color2, color3 : str
            每个特征对应的颜色。
        save_path : str
            图像保存路径（包括文件名）。
        title : str
            图像标题。
        dpi : int
            保存图像的分辨率。
    """
    plt.figure(figsize=(8, 6))
    sns.kdeplot(df[col1], shade=True, cut=0, color=color1, label=label1)
    sns.kdeplot(df[col2], shade=True, cut=0, color=color2, label=label2)
    sns.kdeplot(df[col3], shade=True, cut=0, color=color3, label=label3)

    plt.title(title)
    plt.xlabel('Confidence', fontsize=15)
    plt.ylabel('Density', fontsize=15)
    plt.legend(loc='upper left', fontsize=22) 
    plt.tight_layout()

    # 创建保存目录（如果不存在）
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=dpi)
    plt.close()


def calculate_metrics(y_true, y_pred, y_prob):
    """Calculate overall metrics"""
    accuracy = metrics.accuracy_score(y_true, y_pred)
    precision = metrics.precision_score(y_true, y_pred, average='macro')
    recall = metrics.recall_score(y_true, y_pred, average='macro')
    specificity = []
    for cls in range(len(np.unique(y_true))):
        y_true_bin = (y_true == cls).astype(int)
        y_pred_bin = (y_pred == cls).astype(int)
        tn, fp, fn, tp = metrics.confusion_matrix(y_true_bin, y_pred_bin).ravel()
        spec = tn / (tn + fp)
        specificity.append(spec)
    specificity = np.mean(specificity)
    f1 = metrics.f1_score(y_true, y_pred, average='macro')
    kappa = metrics.cohen_kappa_score(y_true, y_pred)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1': f1,
        'kappa': kappa
    }

def calculate_class_metrics(y_true, y_pred, y_prob):
    """Calculate per-class metrics"""
    n_classes = len(np.unique(y_true))
    class_metrics = []
    
    for cls in range(n_classes):
        y_true_bin = (y_true == cls).astype(int)
        y_pred_bin = (y_pred == cls).astype(int)
        y_prob_bin = y_prob[:, cls]
        
        tn, fp, fn, tp = metrics.confusion_matrix(y_true_bin, y_pred_bin).ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
        auc_score = metrics.roc_auc_score(y_true_bin, y_prob_bin)
        
        fpr, tpr, _ = metrics.roc_curve(y_true_bin, y_prob_bin)
        
        class_metrics.append({
            'class': cls,
            'class_name': class_names[cls],
            'sensitivity': sensitivity,
            'specificity': specificity,
            'precision': precision,
            'f1': f1,
            'auc': auc_score,
            'fpr': fpr,
            'tpr': tpr
        })
    
    return class_metrics

def plot_tsne(features, labels, save_path):
    """Generate and save t-SNE plot"""
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features)
    
    #colors = ['r','g', 'b', 'c', 'm']
    plt.figure(figsize=(10, 8))
    #scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='tab10')
    #plt.legend(fontsize=18)
    #plt.colorbar(scatter)
    #plt.title('t-SNE visualization')
    #plt.savefig(save_path)
    #plt.close()    
    
    colors = ['r','g', 'b', 'c', 'm']
    num_classes = len(class_names)
    for i in range(num_classes):
        idx = np.array(labels) == i
        plt.scatter(features_2d[idx, 0], features_2d[idx, 1], 
                    c=colors[i], label=class_names[i], alpha=0.6)
    #plt.legend(fontsize=18)
    #save_path = os.path.join(tsne_dir, f"tsne_epoch.jpg")
    plt.savefig(save_path)
    plt.close()
    

    
def plot_cosine_similarity(feat1, feat2, save_path=None):
    """
    计算每个样本在两个特征空间下的余弦相似度，并绘制散点图。

    参数:
        feat1: np.ndarray, shape [a, b]
        feat2: np.ndarray, shape [a, b]
        save_path: str or None, 保存路径（如 'cosine_similarity.pdf'）

    返回:
        cos_sims: np.ndarray, shape [a], 每个样本的余弦相似度
    """
    assert feat1.shape == feat2.shape, "两个特征维度必须一致"

    # 归一化特征向量
    norm_feat1 = feat1 / np.linalg.norm(feat1, axis=1, keepdims=True)
    norm_feat2 = feat2 / np.linalg.norm(feat2, axis=1, keepdims=True)

    # 计算每个样本对的余弦相似度（逐行点积）
    cos_sims = np.sum(norm_feat1 * norm_feat2, axis=1)

    # 可视化（x=y 为理想情况）
    sns.set(style="whitegrid", font_scale=1.5)
    plt.figure(figsize=(8, 6))
    x = np.arange(len(cos_sims))  # 样本编号
    y = cos_sims

    plt.scatter(x, y, s=80, alpha=0.8, color="#1f77b4", label='Cosine Similarity')
    plt.plot(x, np.ones_like(x), '--', color='gray', label='Ideal (sim=1)')

    plt.ylim(0, 1.05)
    plt.xlabel("Sample Index", fontsize=16)
    plt.ylabel("Cosine Similarity", fontsize=16)
    #plt.title("Cosine Similarity between Feature 1 and Feature 2", fontsize=20)
    plt.legend(fontsize=22)
    plt.tight_layout()
    sns.despine()

    if save_path:
        plt.savefig(save_path, dpi=300)

def criterion(logit, truth):

    loss = nn.CrossEntropyLoss()(logit, truth)

    return loss

def metric(logit, truth):
    # prob = F.sigmoid(logit)
    _, prediction = torch.max(logit.data, 1)

    acc = torch.sum(prediction == truth)
    return acc

def train(net,train_dataloader,model_name):

    #net.set_mode('train')
    train_loss = 0
    train_dia_acc = 0  
    train_sps_acc = 0
    for index, (clinic_image, derm_image, meta_data, label) in enumerate(train_dataloader):
        opt.zero_grad()
        
        clinic_image = clinic_image.cuda()
        derm_image   = derm_image.cuda()
#         meta_data    = meta_data.cuda()
        
        # Diagostic label
        diagnosis_label = label[1].long().cuda()

        #print()
        logit_diagnosis_fusion, features, confidence_loss, _, _, _, _, _, _, _ = net(
            clinic_image, derm_image, diagnosis_label, update_memory=True
        )
        #print(logit_diagnosis_fusion.shape)
        #print("k_val_test:",k_val_test)
        
        loss_fusion = criterion(logit_diagnosis_fusion, diagnosis_label)           
        #loss_clic = net.criterion(logit_diagnosis_clic, diagnosis_label)
        #loss_derm = net.criterion(logit_diagnosis_derm, diagnosis_label)
        loss = loss_fusion+ confidence_loss
        # loss = loss_fusion
        # loss = confidence_loss

        dia_acc_fusion = torch.true_divide(metric(logit_diagnosis_fusion, diagnosis_label), clinic_image.size(0))
        #dia_acc_clic = torch.true_divide(net.metric(logit_diagnosis_clic, diagnosis_label), clinic_image.size(0))
        #dia_acc_derm = torch.true_divide(net.metric(logit_diagnosis_derm, diagnosis_label), clinic_image.size(0))

        dia_acc = dia_acc_fusion
        #dia_acc = torch.true_divide(dia_acc_fusion + dia_acc_clic + dia_acc_derm, 3)

#         sps_acc_fusion = net.metric(logit_pn_fusion, pn_label)
#         sps_acc_clic = net.metric(logit_pn_clic, pn_label)
#         sps_acc_derm = net.metric(logit_pn_derm, pn_label)

#         sps_acc = torch.true_divide(sps_acc_fusion + sps_acc_clic + sps_acc_derm, 3)


        loss.backward()
        opt.step()

        train_loss += loss.item()
        train_dia_acc += dia_acc.item()
#         train_sps_acc += sps_acc.item()

    train_loss = train_loss / (index + 1) # Because the index start with the value 0f zero
    train_dia_acc = train_dia_acc / (index + 1)
#     train_sps_acc = train_sps_acc / (index + 1)

    return train_loss,train_dia_acc

def validation(net,val_dataloader,model_name, epoch):
    net.eval()
    val_loss = 0
    val_dia_acc = 0
    vaL_sps_acc = 0

    
    all_preds = []
    all_labels = []
    all_probs = []
    all_features = []

    original_missing_all = []
    reconstructed_all = []

    # 收集 TCP confidence 值
    fundus_conf_list = []
    oct_conf_list = []    
    fusion_conf_list = []
    epoch_gate_stats = {
        "missing_total": 0,
        "missing_fundus": 0,
        "missing_oct": 0,
        "gate_activated_total": 0,
        "gate_activated_fundus": 0,
        "gate_activated_oct": 0,
        "retrieval_label_match_top1": 0,
        "retrieval_label_match_top5": 0,
        "retrieval_label_match_top10": 0,
        "retrieval_label_match_topk": 0,
        "retrieval_events": 0,
        "retrieved_k_sum": 0
    }
    all_competition_records = []
    
    for index, (clinic_image, derm_image, meta_data, label) in enumerate(val_dataloader):

        clinic_image = clinic_image.cuda()
        derm_image   = derm_image.cuda()
#         meta_data    = meta_data.cuda()

        diagnosis_label = label[1].long().cuda()

        with torch.no_grad():
          
          
            logits, features, confidence_loss, original_feat, reconstructed_feat, TCPConfidence_fundus, TCPConfidence_oct, Confidence_fusion, gate_stats, competition_records = net(
                clinic_image, derm_image, diagnosis_label, update_memory=False
            )
            #print("k_val_test:",k_val_test)
            #print("gate_decision_print:",gate_decision_print)
           
            #batch_intermediate = net.get_intermediate_features(clinic_image, derm_image)
            #batch_attention = net.get_attention_maps(clinic_image, derm_image)
            
            loss = criterion(logits, diagnosis_label)
            loss = loss+confidence_loss
            # loss = loss
            # loss = confidence_loss
            probs = F.softmax(logits, dim=1)
            _, preds = torch.max(logits.data, 1)
            
            
            acc = torch.true_divide(metric(logits, diagnosis_label), clinic_image.size(0))
            
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(diagnosis_label.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_features.extend(features.cpu().numpy())

            # 提取每个 batch 的 TCP confidence
            fundus_conf_list.extend(TCPConfidence_fundus.view(-1).cpu().numpy())
            oct_conf_list.extend(TCPConfidence_oct.view(-1).cpu().numpy())
            fusion_conf_list.extend(Confidence_fusion.view(-1).cpu().numpy())
            all_competition_records.extend(competition_records)
            for k in epoch_gate_stats.keys():
                epoch_gate_stats[k] += gate_stats.get(k, 0)

            #print(original_feat[0].shape)
            if len(original_feat) != 0 and len(reconstructed_feat) != 0:
                orig_feats = torch.stack(original_feat).cpu().numpy()
                recon_feats = torch.stack(reconstructed_feat).cpu().numpy()
                original_missing_all.append(orig_feats)
                reconstructed_all.append(recon_feats)    
  
        val_loss += loss.item()
        val_dia_acc += acc.item()
#         vaL_sps_acc += sps_acc.item()

    val_loss = val_loss / (index + 1)
    val_dia_acc = val_dia_acc / (index + 1)
    
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_features = np.array(all_features)
    n_classes = all_probs.shape[1]
    
    
    metrics_dict = calculate_metrics(all_labels, all_preds, all_probs)
    class_metrics = calculate_class_metrics(all_labels, all_preds, all_probs)
    
    
    #np.save(f'{out_dir}/features.npy', all_features)
    #np.save(f'{out_dir}/labels.npy', all_labels)
    
    # Confusion Matrix
    # cm = confusion_matrix(all_labels, all_preds)
    # plt.figure(figsize=(10, 8))
    # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
    #             xticklabels=[f'Class {i}' for i in range(n_classes)],
    #             yticklabels=[f'Class {i}' for i in range(n_classes)])
    # plt.xlabel('Predicted')
    # plt.ylabel('True')
    # plt.title('Confusion Matrix')
    # plt.savefig(f'{out_dir}/confusion_matrix_{epoch}.png')
    # plt.close()
    # np.savetxt(f'{out_dir}/confusion_matrix_{epoch}.txt', cm, fmt='%d')
    
    # Save class metrics
    auc_file = f'{out_dir}/class_metrics_{epoch}.txt'
    with open(auc_file, 'w') as f:
        f.write('Class\tName\tAUC\tSensitivity\tSpecificity\tPrecision\tF1\n')
        for cls_metric in class_metrics:
            f.write(f"{cls_metric['class']}\t"
                    f"{cls_metric['class_name']}\t"
                    f"{cls_metric['auc']:.4f}\t"
                    f"{cls_metric['sensitivity']:.4f}\t"
                    f"{cls_metric['specificity']:.4f}\t"
                    f"{cls_metric['precision']:.4f}\t"
                    f"{cls_metric['f1']:.4f}\n")
    
    # Plot and save ROC curves for each class
    plt.figure(figsize=(10, 8))
    colors = ['blue', 'green', 'red', 'cyan', 'magenta']
    
    for cls_metric in class_metrics:
        # Save ROC coordinates to file
        roc_file = f"{out_dir}/roc_{cls_metric['class_name']}_{epoch}.txt"
        with open(roc_file, 'w') as f:
            f.write(f"Class {cls_metric['class']} ({cls_metric['class_name']}) ROC Coordinates:\n")
            f.write("FPR,TPR\n")
            for fpr, tpr in zip(cls_metric['fpr'], cls_metric['tpr']):
                f.write(f"{fpr:.6f},{tpr:.6f}\n")
            f.write(f"AUC: {cls_metric['auc']:.6f}\n")
        
        # Plot ROC curve
        plt.plot(cls_metric['fpr'], cls_metric['tpr'], 
                 color=colors[cls_metric['class']], 
                 lw=2, 
                 label=f"{cls_metric['class_name']} (AUC = {cls_metric['auc']:.2f})")
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Each Class')
    plt.legend(loc="lower right")
    plt.legend(fontsize=24)
    plt.savefig(f'{out_dir}/roc_classes_{epoch}.png')
    plt.close()
    
    # Plot t-SNE
    #print(all_features)
    # plot_tsne(all_features, all_labels, f'{out_dir}/tsne_{epoch}.png')

    #相关性散点图
    # print(original_missing_all[0].shape)
    # original_missing_all = np.array(original_missing_all)
    
    # reconstructed_all = np.array(reconstructed_all)
    
    if len(original_missing_all) > 0 and len(reconstructed_all) > 0:
        original_missing_all = np.concatenate(original_missing_all, axis=0)
        reconstructed_all = np.concatenate(reconstructed_all, axis=0)
        print(original_missing_all.shape)

        plot_cosine_similarity(
            feat1=original_missing_all,
            feat2=reconstructed_all,
            save_path=f"{out_dir}/feature_correlation_epoch_{epoch}.png"
        )

    #plot_kde_and_save
    fundus_conf = np.array(fundus_conf_list)
    oct_conf = np.array(oct_conf_list)
    fusion_conf = np.array(fusion_conf_list)
    print("Fundus conf min/max:", fundus_conf.min(), fundus_conf.max())
    print("OCT   conf min/max:", oct_conf.min(), oct_conf.max())
    print("Fusion conf min/max:", fusion_conf.min(), fusion_conf.max())

    conf_df = pd.DataFrame({
        "Clinic": fundus_conf_list,
        "Derm": oct_conf_list,
        "Fusion": fusion_conf_list
    })

    plot_kde_three_and_save(conf_df,
                        col1='Clinic',
                        col2='Derm',
                        col3='Fusion',
                        label1='Confidence - Clinic',
                        label2='Confidence - Derm',
                        label3='Confidence - Fusion',
                        color1='r',
                        color2='b',
                        color3='g',
                        save_path=f'{out_dir}/kde_confidence_epoch_{epoch}.png',
                        title=f'Uncertainty KDE Plot')

    plot_reliability_diagram_with_boxplot(conf_df['Fusion'], all_preds, all_labels,
        save_path_reliability=f'{out_dir}/calibration_confidence_Fusion_epoch_{epoch}.png', 
        save_path_boxplot=f'{out_dir}/calibration_gap_box_Fusion_epoch_{epoch}.png')

    plot_reliability_diagram_with_boxplot(conf_df['Clinic'], all_preds, all_labels,
        save_path_reliability=f'{out_dir}/calibration_confidence_Clinic_epoch_{epoch}.png', 
        save_path_boxplot=f'{out_dir}/calibration_gap_box_Clinic_epoch_{epoch}.png')

    plot_reliability_diagram_with_boxplot(conf_df['Derm'], all_preds, all_labels,
        save_path_reliability=f'{out_dir}/calibration_confidence_Derm_epoch_{epoch}.png', 
        save_path_boxplot=f'{out_dir}/calibration_gap_box_Derm_epoch_{epoch}.png')

    # ===== 新增1：门控激活统计（每个epoch） =====
    missing_total = max(epoch_gate_stats["missing_total"], 1)
    missing_fundus = max(epoch_gate_stats["missing_fundus"], 1)
    missing_oct = max(epoch_gate_stats["missing_oct"], 1)
    gate_summary = {
        "epoch": int(epoch),
        "missing_total": int(epoch_gate_stats["missing_total"]),
        "missing_fundus": int(epoch_gate_stats["missing_fundus"]),
        "missing_oct": int(epoch_gate_stats["missing_oct"]),
        "gate_activated_total": int(epoch_gate_stats["gate_activated_total"]),
        "gate_activated_fundus": int(epoch_gate_stats["gate_activated_fundus"]),
        "gate_activated_oct": int(epoch_gate_stats["gate_activated_oct"]),
        "gate_activation_ratio_total": float(epoch_gate_stats["gate_activated_total"] / missing_total),
        "gate_activation_ratio_fundus": float(epoch_gate_stats["gate_activated_fundus"] / missing_fundus),
        "gate_activation_ratio_oct": float(epoch_gate_stats["gate_activated_oct"] / missing_oct),
    }
    with open(f'{out_dir}/gate_stats_epoch_{epoch}.txt', 'w') as f:
        for k, v in gate_summary.items():
            f.write(f'{k}: {v}\n')
    log.write('\n[Gate Stats - Validation]\n')
    for k, v in gate_summary.items():
        log.write(f'{k}: {v}\n')

    # ===== 新增2：memory 检索标签一致性 top-k / top-1 =====
    retrieval_events = max(epoch_gate_stats["retrieval_events"], 1)
    retrieval_summary = {
        "epoch": int(epoch),
        "retrieval_events": int(epoch_gate_stats["retrieval_events"]),
        "avg_selected_k": float(epoch_gate_stats["retrieved_k_sum"] / retrieval_events),
        "top1_label_consistency": float(epoch_gate_stats["retrieval_label_match_top1"] / retrieval_events),
        "top5_label_consistency": float(epoch_gate_stats["retrieval_label_match_top5"] / retrieval_events),
        "top10_label_consistency": float(epoch_gate_stats["retrieval_label_match_top10"] / retrieval_events),
        "topk_label_consistency": float(epoch_gate_stats["retrieval_label_match_topk"] / retrieval_events),
        "top1_label_match_count": int(epoch_gate_stats["retrieval_label_match_top1"]),
        "top5_label_match_count": int(epoch_gate_stats["retrieval_label_match_top5"]),
        "top10_label_match_count": int(epoch_gate_stats["retrieval_label_match_top10"]),
        "topk_label_match_count": int(epoch_gate_stats["retrieval_label_match_topk"]),
    }
    with open(f'{out_dir}/retrieval_label_consistency_epoch_{epoch}.txt', 'w') as f:
        for k, v in retrieval_summary.items():
            f.write(f'{k}: {v}\n')
    log.write('\n[Retrieval Label Consistency - Validation]\n')
    for k, v in retrieval_summary.items():
        log.write(f'{k}: {v}\n')

    # ===== 新增3：案例分析（单模态 vs 融合模态竞争） =====
    case_lines = []
    fusion_win_case = None
    clinic_win_case = None
    derm_win_case = None
    for rec in all_competition_records:
        y = rec["label"]
        clinic_ok = int(rec["pred_clinic"] == y)
        derm_ok = int(rec["pred_derm"] == y)
        fusion_ok = int(rec["pred_fusion"] == y)
        if fusion_win_case is None and fusion_ok == 1 and (clinic_ok == 0 or derm_ok == 0):
            fusion_win_case = rec
        if clinic_win_case is None and clinic_ok == 1 and derm_ok == 0 and fusion_ok == 0:
            clinic_win_case = rec
        if derm_win_case is None and derm_ok == 1 and clinic_ok == 0 and fusion_ok == 0:
            derm_win_case = rec

    def _fmt_case(name, rec):
        if rec is None:
            return f'{name}: not found in this epoch\n'
        return (
            f"{name}: label={rec['label']}, pred_clinic={rec['pred_clinic']}, "
            f"pred_derm={rec['pred_derm']}, pred_fusion={rec['pred_fusion']}, "
            f"conf_clinic={rec['conf_clinic']:.4f}, conf_derm={rec['conf_derm']:.4f}, "
            f"conf_fusion={rec['conf_fusion']:.4f}\n"
        )

    case_lines.append(f'epoch: {epoch}\n')
    case_lines.append(_fmt_case('case_fusion_win', fusion_win_case))
    case_lines.append(_fmt_case('case_clinic_win', clinic_win_case))
    case_lines.append(_fmt_case('case_derm_win', derm_win_case))
    with open(f'{out_dir}/competition_case_epoch_{epoch}.txt', 'w') as f:
        f.writelines(case_lines)
    log.write('\n[Competition Case Study - Validation]\n')
    for line in case_lines:
        log.write(line)
    

    log.write('\nValidation Metrics:\n')
    for metric_name, value in metrics_dict.items():
        log.write(f'{metric_name}: {value:.4f}\n')

    return val_loss,val_dia_acc


def run_train(model_name,mode,i):
    log.write('** start training here! **\n')
    #best_acc = 0
    es = 0
    patience = 50
    best_mean_acc = 0 
    best_loss = 300
    
    for epoch in range(epochs):
        swa_lr = cosine_learning_schule[epoch]
        adjust_learning_rate(opt, swa_lr)

        # train_mode
        train_loss,train_dia_acc = train(net, train_dataloader,model_name)
        log.write('Round: {}, epoch: {}, Train Loss: {:.4f}, Train Dia Acc: {:.4f}\n'.format(i, epoch, train_loss,
                                                                                                         train_dia_acc
                                                                                                         ))

        # validation mode
        val_loss,val_dia_acc = validation(net, val_dataloader,model_name, epoch)
        
        val_acc = val_dia_acc
        val_mean_acc = val_dia_acc
        
        log.write('Round: {}, epoch: {}, Valid Loss: {:.4f}, Valid Dia Acc: {:.4f}\n'.format(i, epoch, val_loss,
                                                                                                         val_dia_acc
                                                                                                         ))

     
        if val_mean_acc > best_mean_acc:
            es = 0
            best_mean_acc = val_mean_acc
            #torch.save(net, out_dir + '/checkpoint/{diag_label_guided_gating}_best_model.pth')
            log.write('Current Best Mean Acc is {}'.format(best_mean_acc))
        #  else:
        #      es += 1
        #      print("Counter {} of {}".format(es,patience))
          
        #      if es > patience:
        #          print("Early stopping with best_mean_acc: {:.4f}".format(best_mean_acc), "and val_mean_acc for this epoch: {:.4f}".format(val_mean_acc))
        #          break
  
        #if epoch == 150:
            #torch.save(net, out_dir + '/checkpoint/{diag_label_guided_gating}_model.pth')
        if epoch > (epochs - swa_epoch) and epoch % 1 == 0:
            opt.update_swa()
            log.write('SWA Epoch: {}'.format(epoch))

    #torch.save(net, out_dir+'/swa_{}_resnet50_model.pth')

        
if __name__ == '__main__':
    # Hyperparameters
    
    mode = 'multimodal'
    model_name = 'Our-MMC_r1_missing=0.6_complete_experiments_lr_1e-5_seed_42_update_memory_bank'
    # model_name = 'Our-SPC_test_image'
    shape = (224, 224)
    batch_size = 32
    num_workers = 8
    data_mode = 'Normal'
    deterministic = True
    if deterministic:
        if data_mode == 'Normal':
          random_seeds = 42
        elif data_mode == 'self_evaluated':
          random_seeds = 183
    rounds = 1
    lr = 1e-5
    epochs = 250
    swa_epoch = 50

    train_dataloader, val_dataloader = generate_dataloader(shape, batch_size, num_workers, data_mode)
    
    for i in range(rounds):
        if deterministic:
            set_seed(random_seeds + i)
      # create logger
        print(random_seeds+i)
        log, out_dir = CraateLogger(mode, model_name,i,data_mode)
        net = Base_Model(num_classes=4, p_missing=0.6, memory_size=500).cuda()
        #net = net.to('cuda:0')
        #net.initialize_memory(train_dataloader)
      # create optimizer
        optimizer = optim.Adam(net.parameters(), lr=lr)
        opt = SWA(optimizer)
      # create learning schdule
        cosine_learning_schule = create_cosine_learing_schdule(epochs, lr)
        run_train(model_name,mode,i)
