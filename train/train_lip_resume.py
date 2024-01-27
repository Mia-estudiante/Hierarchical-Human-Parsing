import argparse
import os
import random
import sys
import time

# sys.path.append('/data/Hierarchical-Human-Parsing')

import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.nn import functional as F
from torch.nn.parallel.scatter_gather import gather, scatter
from torch.utils import data
from dataset.data_lip import DatasetGenerator
#################################################
#수정1. 기존에는 baseline 모듈에서 get_model 함수를 불러왔다.
from network.gnn_parse import get_model 
# from network.baseline import get_model       
#################################################
from utils.lovasz_loss import ABRLovaszCELoss as ABRLovaszLoss 
from utils.metric import *
from utils.parallel import DataParallelModel, DataParallelCriterion
from utils.visualize import inv_preprocess, decode_predictions

#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

METHOD = 'magnet_stem_grid_st1fea_gate_ob'
TRAIN_CS_PATH = "/data/Hierarchical-Human-Parsing/data/LIP/train_set"
VAL_CS_PATH = "/data/Hierarchical-Human-Parsing/data/LIP/val_set"
TRAIN_LST_CS_PATH = "/data/Hierarchical-Human-Parsing/data/LIP/train_id.txt"
VAL_LST_CS_PATH = "/data/Hierarchical-Human-Parsing/data/LIP/val_id.txt"

#############


'''
hyperparameters
- crop size
- batch size
- save log
- number of classes
- learning rate
- learning mode
- ignore label(255)
- init
- number of save classes
- hbody-cls
'''

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Segmentation')
    parser.add_argument('--method', type=str, default=METHOD)
    # Datasets
    parser.add_argument('--root', default=TRAIN_CS_PATH, type=str)
    parser.add_argument('--val-root', default=VAL_CS_PATH, type=str)
    parser.add_argument('--lst', default=TRAIN_LST_CS_PATH, type=str)
    parser.add_argument('--val-lst', default=VAL_LST_CS_PATH, type=str)
    parser.add_argument('--crop-size', type=int, default=CROP)
    parser.add_argument('--num-classes', type=int, default=CLASSES)
    parser.add_argument('--hbody-cls', type=int, default=HBODY_CLS)
    parser.add_argument('--fbody-cls', type=int, default=FBODY_CLS)
    # Optimization options
    parser.add_argument('--epochs', default=EPOCHS, type=int)
    parser.add_argument('--batch-size', default=BS, type=int)
    parser.add_argument('--learning-rate', default=LR, type=float)
    parser.add_argument('--lr-mode', type=str, default=LR_MODE)
    parser.add_argument('--ignore-label', type=int, default=IGNORE_LABEL)
    # Checkpoints
    parser.add_argument('--restore-from', default=CHECKPOINT, type=str)
    parser.add_argument('--snapshot_dir', type=str, default=SNAPSHOT_FROM)
    parser.add_argument('--log-dir', type=str, default=LOG_DIR)
    parser.add_argument('--init', action="store_true", default=INIT)
    parser.add_argument('--save-num', type=int, default=SAVE_NUM)
    # Misc
    parser.add_argument('--seed', type=int, default=SEED)
    args = parser.parse_args()

    return args


def adjust_learning_rate(optimizer, epoch, i_iter, iters_per_epoch, method='poly'):
    if method == 'poly':
        current_step = epoch * iters_per_epoch + i_iter
        max_step = args.epochs * iters_per_epoch
        lr = args.learning_rate * ((1 - current_step / max_step) ** 0.9)
    else:
        lr = args.learning_rate
    optimizer.param_groups[0]['lr'] = lr
    return lr

def main(args):
    # initialization
    print(torch.cuda.device_count())
    print("Input arguments:")
    for key, val in vars(args).items():
        print("{:16} {}".format(key, val))

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)
    writer = SummaryWriter(log_dir=os.path.join(args.log_dir, args.method))

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    cudnn.benchmark = True

    #Step1. 사용할 모델 설정
    # conduct seg network
    seg_model = get_model(num_classes=args.num_classes)                     #1. 사용할 모델 구조 제작

    saved_state_dict = torch.load(args.restore_from) #pretrained weight     #2. 모델에 입힐 weight 파일 불러오기
    
    ####재시작!###########################################
    if args.resume_from:
        seg_model.load_state_dict(saved_state_dict)
        start_epoch = 3
        end_epoch = 101
        print(f"{start_epoch} epoch까지 학습된 모델 재시작, {end_epoch-start_epoch} epoch를 추가 학습!")
        print(f"{args.restore_from} 에서 불러온 model weight file...")
    ####재시작!###########################################

    # new_params = seg_model.state_dict().copy()       #after initalized  

    # if 'state_dict' in saved_state_dict:
    #     saved_state_dict = saved_state_dict['state_dict']

    # for i in saved_state_dict: #pretrained weight                           #3. 해당 모델은 backbone을 초기화하는데 ImageNet으로 pretrained된 ResNet을 사용
    #     try:                                                                #   저자는 ResNet weight file을 제공하지 않았기에, 임의로 이름과 dimension을 맞춰줘야 한다.
    #         i_parts = i.split('.')                                          #   이름과 dimension을 확인하는 코든는 compare_resnet.py 를 참고하자.
    #         if not i_parts[0] == 'fc':
    #             seg_model.state_dict()['encoder.' + '.'.join(i_parts[:])].copy_(saved_state_dict[i])
    #     except:
    #             print(i, seg_model.state_dict()['encoder.' + '.'.join(i_parts[:])].size(), saved_state_dict[i].size())
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(seg_model)
        # model = DataParallelModel(seg_model) #seg_model 그대로 사용        #4. multi-gpu를 사용하지 않는 방향으로 설정               
    else:
        model = seg_model
    model.float()
    model.cuda()
    
    #Step2. DataLoader 설정
    # define dataloader
    train_loader = data.DataLoader(DatasetGenerator(root=args.root, list_path=args.lst,
                                                    crop_size=args.crop_size, training=True),
                                   batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = data.DataLoader(DatasetGenerator(root=args.val_root, list_path=args.val_lst,
                                                  crop_size=args.crop_size, training=False),
                                 batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    #Step3. loss function과 Optimizer 설정
    # define criterion & optimizer
    criterion = ABRLovaszLoss(ignore_index=args.ignore_label, only_present=True).cuda()
    # criterion = DataParallelCriterion(criterion).cuda()

    optimizer = optim.SGD(
        [{'params': filter(lambda p: p.requires_grad, seg_model.parameters()), 'lr': args.learning_rate}],
        lr=args.learning_rate, momentum=0.9, weight_decay=5e-4)

    # key points
    best_val_mIoU = 0
    best_val_pixAcc = 0
    start = time.time()

    for epoch in range(start_epoch+1, end_epoch):
        epoch_start = time.time()
        print('\n{} | {}'.format(epoch, args.epochs - 1))
        #Step4. Train
        # training
        _ = train(model, train_loader, epoch, criterion, optimizer, writer)
        
        #Step5. Valid
        # validation
        if epoch %2 ==0 or epoch > args.epochs-5:
            val_pixacc, val_miou = validation(model, val_loader, epoch, writer)
            # save model
            if val_pixacc > best_val_pixAcc:
                best_val_pixAcc = val_pixacc
            if val_miou > best_val_mIoU:
                best_val_mIoU = val_miou
                model_dir = os.path.join(args.snapshot_dir, args.method+ '_epoch'+ str(epoch)+'_miou.pth')
                torch.save(seg_model.state_dict(), model_dir)
                print('Model saved to %s' % model_dir)
        epoch_end = time.time()
        wf_train_epoch = open(os.path.join(args.snapshot_dir, "train_epoch.txt"), "a")
        wf_train_epoch.write(f"1epoch 소요시간: {epoch_start}-{epoch_end}\n")

    os.rename(model_dir, os.path.join(args.snapshot_dir, args.method + '_miou'+str(best_val_mIoU)+'.pth'))
    print('Complete using', time.time() - start, 'seconds')
    print('Best pixAcc: {} | Best mIoU: {}'.format(best_val_pixAcc, best_val_mIoU))


def train(model, train_loader, epoch, criterion, optimizer, writer):
    # set training mode
    model.train()
    train_loss = 0.0
    iter_num = 0

    # Iterate over data.
    from tqdm import tqdm
    tbar = tqdm(train_loader)
    for i_iter, batch in enumerate(tbar):
        sys.stdout.flush()
        start_time = time.time()
        iter_num += 1
        # adjust learning rate
        iters_per_epoch = len(train_loader)
        lr = adjust_learning_rate(optimizer, epoch, i_iter, iters_per_epoch, method=args.lr_mode)
        image, label, hlabel, flabel, _ = batch
        images, labels, hlabel, flabel = image.cuda(), label.long().cuda(), hlabel.cuda(), flabel.cuda()
        torch.set_grad_enabled(True)

        # zero the parameter gradients
        optimizer.zero_grad()

        # compute output loss
        # print(torch.cuda.device_count())
        ########################
        labels[labels>20] = 0
        hlabel[hlabel>20] = 0
        flabel[flabel>20] = 0
        ########################
        preds = model(images)
        loss = criterion(preds[0], [labels, hlabel, flabel])  # batch mean
        train_loss += loss.item()
        # labels = torch.randint(0, 7, (2, 473, 473)).cuda()
        # flabel = torch.randint(0, 2, (2, 473, 473)).cuda() 
        # hlabel = torch.randint(0, 3, (2, 473, 473)).cuda()
        
        # compute gradient and do SGD step
        loss.backward()
        optimizer.step()

        if i_iter % 10 == 0:
            writer.add_scalar('learning_rate', lr, iter_num + epoch * len(train_loader))
            writer.add_scalar('train_loss', train_loss / iter_num, iter_num + epoch * len(train_loader))

        batch_time = time.time() - start_time
        # plot progress
        tbar.set_description('{} / {} | Time: {batch_time:.4f} | Loss: {loss:.4f}'.format(iter_num, len(train_loader),
                                                                                  batch_time=batch_time,
                                                                                  loss=train_loss / iter_num))
        wf_train = open(os.path.join(args.snapshot_dir, "train_log.txt"), "a")
        wf_train.write('{} / {} | Time: {batch_time:.4f} | Loss: {loss:.4f}\n'.format(iter_num, len(train_loader),
                                                                                  batch_time=batch_time,
                                                                                  loss=train_loss / iter_num))

        wf_train.close()

    epoch_loss = train_loss / iter_num
    writer.add_scalar('train_epoch_loss', epoch_loss, epoch)
    tbar.close()

    return epoch_loss


def validation(model, val_loader, epoch, writer):
    # set evaluate mode
    model.eval()

    total_correct, total_label = 0, 0
    total_correct_hb, total_label_hb = 0, 0
    total_correct_fb, total_label_fb = 0, 0
    hist = np.zeros((args.num_classes, args.num_classes))
    hist_hb = np.zeros((args.num_classes, args.num_classes))
    hist_fb = np.zeros((args.num_classes, args.num_classes))
    # hist_hb = np.zeros((args.hbody_cls, args.hbody_cls))
    # hist_fb = np.zeros((args.fbody_cls, args.fbody_cls))

    # Iterate over data.
    from tqdm import tqdm
    tbar = tqdm(val_loader)
    for idx, batch in enumerate(tbar):
        image, target, hlabel, flabel, _ = batch
        image, target, hlabel, flabel = image.cuda(), target.cuda(), hlabel.cuda(), flabel.cuda()

        with torch.no_grad():
            h, w = target.size(1), target.size(2)
            ########################
            hlabel[hlabel>20] = 0
            flabel[flabel>20] = 0
            ########################
            outputs = model(image)
            scattered_tensors = scatter(outputs[0], [0], dim=0)  # Example devices
            outputs = gather(scattered_tensors, 0, dim=0)
            ########################
            preds = F.interpolate(input=outputs[0], size=(h, w), mode='bilinear', align_corners=True)
            preds_hb = F.interpolate(input=outputs[1], size=(h, w), mode='bilinear', align_corners=True)
            preds_fb = F.interpolate(input=outputs[2], size=(h, w), mode='bilinear', align_corners=True)
            if idx % 50 == 0:
                img_vis = inv_preprocess(image, num_images=args.save_num)
                label_vis = decode_predictions(target.int(), num_images=args.save_num, num_classes=args.num_classes)
                pred_vis = decode_predictions(torch.argmax(preds, dim=1), num_images=args.save_num,
                                              num_classes=args.num_classes)

                # visual grids
                img_grid = torchvision.utils.make_grid(torch.from_numpy(img_vis.transpose(0, 3, 1, 2)))
                label_grid = torchvision.utils.make_grid(torch.from_numpy(label_vis.transpose(0, 3, 1, 2)))
                pred_grid = torchvision.utils.make_grid(torch.from_numpy(pred_vis.transpose(0, 3, 1, 2)))
                writer.add_image('val_images', img_grid, epoch * len(val_loader) + idx + 1)
                writer.add_image('val_labels', label_grid, epoch * len(val_loader) + idx + 1)
                writer.add_image('val_preds', pred_grid, epoch * len(val_loader) + idx + 1)

            # pixelAcc
            correct, labeled = batch_pix_accuracy(preds.data, target)
            correct_hb, labeled_hb = batch_pix_accuracy(preds_hb.data, hlabel)
            correct_fb, labeled_fb = batch_pix_accuracy(preds_fb.data, flabel)
            # mIoU
            hist += fast_hist(preds, target, args.num_classes)
            hist_hb += fast_hist(preds_hb, hlabel, args.num_classes)
            hist_fb += fast_hist(preds_fb, flabel, args.num_classes)
            # hist_hb += fast_hist(preds_hb, hlabel, args.hbody_cls)
            # hist_fb += fast_hist(preds_fb, flabel, args.fbody_cls)

            total_correct += correct
            total_correct_hb += correct_hb
            total_correct_fb += correct_fb
            total_label += labeled
            total_label_hb += labeled_hb
            total_label_fb += labeled_fb
            pixAcc = 1.0 * total_correct / (np.spacing(1) + total_label)
            IoU = round(np.nanmean(per_class_iu(hist)) * 100, 2)
            pixAcc_hb = 1.0 * total_correct_hb / (np.spacing(1) + total_label_hb)
            IoU_hb = round(np.nanmean(per_class_iu(hist_hb)) * 100, 2)
            pixAcc_fb = 1.0 * total_correct_fb / (np.spacing(1) + total_label_fb)
            IoU_fb = round(np.nanmean(per_class_iu(hist_fb)) * 100, 2)
            # plot progress
            tbar.set_description('{} / {} | {pixAcc:.4f}, {IoU:.4f} |' \
                         '{pixAcc_hb:.4f}, {IoU_hb:.4f} |' \
                         '{pixAcc_fb:.4f}, {IoU_fb:.4f}'.format(idx + 1, len(val_loader), pixAcc=pixAcc, IoU=IoU,pixAcc_hb=pixAcc_hb, IoU_hb=IoU_hb,pixAcc_fb=pixAcc_fb, IoU_fb=IoU_fb))
            wf_val = open(os.path.join(args.snapshot_dir, "val_log.txt"), "a")
            wf_val.write('{} / {} | {pixAcc:.4f}, {IoU:.4f} |' \
                         '{pixAcc_hb:.4f}, {IoU_hb:.4f} |' \
                         '{pixAcc_fb:.4f}, {IoU_fb:.4f}\n'.format(idx + 1, len(val_loader), pixAcc=pixAcc, IoU=IoU,pixAcc_hb=pixAcc_hb, IoU_hb=IoU_hb,pixAcc_fb=pixAcc_fb, IoU_fb=IoU_fb))

            wf_val.close()

    print('\n per class iou part: {}'.format(per_class_iu(hist)*100))
    print('per class iou hb: {}'.format(per_class_iu(hist_hb)*100))
    print('per class iou fb: {}'.format(per_class_iu(hist_fb)*100))

    mIoU = round(np.nanmean(per_class_iu(hist)) * 100, 2)
    mIoU_hb = round(np.nanmean(per_class_iu(hist_hb)) * 100, 2)
    mIoU_fb = round(np.nanmean(per_class_iu(hist_fb)) * 100, 2)

    writer.add_scalar('val_pixAcc', pixAcc, epoch)
    writer.add_scalar('val_mIoU', mIoU, epoch)
    writer.add_scalar('val_pixAcc_hb', pixAcc_hb, epoch)
    writer.add_scalar('val_mIoU_hb', mIoU_hb, epoch)
    writer.add_scalar('val_pixAcc_fb', pixAcc_fb, epoch)
    writer.add_scalar('val_mIoU_fb', mIoU_fb, epoch)
    tbar.close()

    wf_val = open(os.path.join(args.snapshot_dir, "val_log.txt"), "a")
    wf_val.write('per class iou part: {}\n'.format(per_class_iu(hist)*100))
    wf_val.write('per class iou hb: {}\n'.format(per_class_iu(hist_hb)*100))
    wf_val.write('per class iou fb: {}\n'.format(per_class_iu(hist_fb)*100))
    wf_val.close()

    return pixAcc, mIoU


if __name__ == '__main__':
    args = parse_args()
    main(args)