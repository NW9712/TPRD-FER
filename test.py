from sklearn.metrics import confusion_matrix
import torch
import argparse
import os
import numpy as np
import utils
from warnings import filterwarnings
from utils.utils_config import get_config
from utils.utils_device import get_device
from sklearn.metrics import balanced_accuracy_score
from fvcore.nn import FlopCountAnalysis, parameter_count_table
import clip
from utils.utils_model import get_model
import matplotlib.pyplot as plt
filterwarnings('ignore')

try:
    rank = int(os.environ["RANK"])
except KeyError:
    rank = 0

def draw_confusion_matrix(label_true, label_pred, label_name, normlize, title="Confusion Matrix", pdf_save_path=None, dpi=100):
    """

    @param label_true: 真实标签，比如[0,1,2,7,4,5,...]
    @param label_pred: 预测标签，比如[0,5,4,2,1,4,...]
    @param label_name: 标签名字，比如['cat','dog','flower',...]
    @param normlize: 是否设元素为百分比形式
    @param title: 图标题
    @param pdf_save_path: 是否保存，是则为保存路径pdf_save_path=xxx.png | xxx.pdf | ...等其他plt.savefig支持的保存格式
    @param dpi: 保存到文件的分辨率，论文一般要求至少300dpi
    @return:

    example：
            draw_confusion_matrix(label_true=y_gt,
                          label_pred=y_pred,
                          label_name=["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"],
                          normlize=True,
                          title="Confusion Matrix on Fer2013",
                          pdf_save_path="Confusion_Matrix_on_Fer2013.png",
                          dpi=300)

    """
    cm1=confusion_matrix(label_true, label_pred)
    cm = confusion_matrix(label_true, label_pred)
    if normlize:
        row_sums = np.sum(cm, axis=1)
        cm = cm / row_sums[:, np.newaxis]
    cm = cm*100.0
    cm1 = cm1*100.0
    plt.imshow(cm, cmap='Blues')
    cm=cm.T
    cm1=cm1.T
    # plt.title(title)
    # plt.xlabel("Predict label")
    # plt.ylabel("Truth label")
    plt.yticks(range(label_name.__len__()), label_name)
    plt.xticks(range(label_name.__len__()), label_name, rotation=45)

    plt.tight_layout()

    plt.colorbar()

    for i in range(label_name.__len__()):
        for j in range(label_name.__len__()):
            color = (1, 1, 1) if i == j else (0, 0, 0)	# 对角线字体白色，其他黑色
            value = float(format('%.2f' % (cm[i, j])))
            # value1=str(value)+'%\n'+str(cm1[i, j])
            value1 = str(value)
            plt.text(i, j, value1, verticalalignment='center', horizontalalignment='center', color=color)

    # plt.show()
    if not pdf_save_path is None:
        plt.savefig(pdf_save_path, bbox_inches='tight',dpi=dpi)


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser(description="Distributed Arcface Training in Pytorch")
    parser.add_argument("--config", type=str, default='configs/fer2013_rn50_TPRD_baseline.py',help="py config file")
    parser.add_argument("--saveroot", type=str,default='/home/niewei/pythonWork/FER/results_ouput/TPRA',help="py config file")
    parser.add_argument("--confusion_matrix_name", type=str,default='Confusion_Matrix',help="Confusion_Matrix")

    # rafdb affect7 affect8 ferplus fer2013
    # _vit_b_16_TPRD_ _rn50_TPRD_
    # baseline disentangle

    args = parser.parse_args()
    cfg = get_config(args.config)

    CLIP_model, _ = clip.load(cfg.clip_model, device='cpu')
    model = get_model(cfg, clip=CLIP_model)

    device = get_device(cfg)
    torch.cuda.set_device(device)

    checkpoint = torch.load(cfg.checkpoints, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)

    print(parameter_count_table(model))

    model.to(device)

    dataloaders=utils.dataloaders.__dict__[cfg.dataset + '_dataloader'](cfg)
    testloader =  dataloaders.run(mode='test')

    with torch.no_grad():
        model.eval()

        model.class_embs = model.text_encoder(model.expression_prompts_learner(),model.tokenized_expression_prompts)
        model.region_embs = model.text_encoder(model.region_prompts_learner(), model.tokenized_region_prompts)
        tensor = torch.rand(1, 3, 224, 224).to(device)
        flops  = FlopCountAnalysis(model, tensor)
        print("FLOPs: ", flops.total()/10**9, '(G)')

        correct_sum = 0
        baccs = []
        labels = []
        preds = []
        for (imgs, targets) in testloader:
            imgs, targets = imgs.to(device), targets.to(device)
            logits, _= model(imgs)

            if cfg.reverse:
                _, predicts = torch.min(logits, 1)
            else:
                _, predicts = torch.max(logits, 1)
            labels.append(targets.cpu().detach().numpy())
            preds.append(predicts.cpu().detach().numpy())
            correct_num = torch.eq(predicts, targets).sum()
            correct_sum += correct_num.cpu()
            baccs.append(balanced_accuracy_score(targets.cpu().numpy(), predicts.cpu().numpy()))

        acc = correct_sum.float() / float(testloader.dataset.__len__())
        acc = np.around(acc.numpy(), 4)
        bacc = np.around(np.mean(baccs), 4)
        labels = np.concatenate(labels,axis=0)
        preds = np.concatenate(preds, axis=0)
        cm = confusion_matrix(labels, preds)
        row_sums = np.sum(cm, axis=1)
        cm = cm / row_sums[:, np.newaxis]
        eyes_matrix = np.eye(len(cfg.expression_prompts))
        mean_acc = np.around(np.sum(cm * eyes_matrix) / len(cfg.expression_prompts), 4)

        print('acc:' + str(acc))
        print('mean acc:' + str(mean_acc))
        print('balanced acc:' + str(bacc))


        expressions_name = cfg.expression_prompts

        if args.confusion_matrix_name:
            draw_confusion_matrix(label_true=labels,
                                  label_pred=preds,
                                  label_name=expressions_name,
                                  normlize=True,
                                  title="Confusion Matrix",
                                  pdf_save_path=os.path.join(os.path.dirname(cfg.checkpoints),args.confusion_matrix_name),
                                  dpi=600)