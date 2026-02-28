import math
import argparse
import sys
sys.path.append("../../meshcnn")
import numpy as np
import pickle, gzip
import os
import shutil
import logging
from collections import OrderedDict
from tqdm import tqdm
import sys
from sklearn.metrics import mean_squared_error
import pandas as pd
from loader import ClimateSegLoader
from model import  Multi_uunet
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize

from torch.nn.parallel import DistributedDataParallel as ddp
from torch.utils.data.distributed import DistributedSampler as ds
import torch.distributed as distri

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

element_list=[3,5,11,12,13,14,19,20,21,22,23,24,26,29,30,31,32,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84]

#解除显存使用限制
use_cuda = torch.cuda.is_available() #not args.no_cuda and torch.cuda.is_available()

device = torch.device("cuda" if use_cuda else "cpu")
# mean_list=np.load('mean_list.npy')
# std_list=np.load('std_list.npy')
# mean_list_X=np.load('mean_list_X.npy')
# std_list_X=np.load('std_list_X.npy')

tgt_device = device

def find_top_10_indices_advanced(lst):
    """
    使用numpy（如果可用）
    """
    try:
        import numpy as np
        # print("\n方法2 - 使用numpy：")
        arr = np.array(lst)
        # 获取最大的10个元素的索引
        top_10_indices = np.argpartition(arr, -10)[-10:]
        # 按值降序排序这些索引
        top_10_indices_sorted = top_10_indices[np.argsort(arr[top_10_indices])][::-1]
        
        # print("最大的10个元素及其序号：")
        for i, idx in enumerate(top_10_indices_sorted, 1):
            print(f"第{i}大: 值={lst[idx]}, 序号={idx}")
            
        return [idx for idx in top_10_indices_sorted]
    except ImportError:
        # print("\n方法2 - numpy未安装，跳过此方法")
        return None
    
def save_checkpoint(state, is_best, epoch, output_folder, filename, logger):
    if epoch > 1 and os.path.exists(output_folder + filename + '_%03d' % (epoch-1) + '.pth.tar'):
        os.remove(output_folder + filename + '_%03d' % (epoch-1) + '.pth.tar')
    torch.save(state, output_folder + filename + '_%03d' % epoch + '.pth.tar')
    if is_best:
        logger.info("Saving new best model")
        shutil.copyfile(output_folder + filename + '_%03d' % epoch + '.pth.tar',
                        output_folder + filename + '_best.pth.tar')

def average_precision(score_cls, true_cls, nclass=3):
    score = score_cls.cpu().numpy()
    true = label_binarize(true_cls.cpu().numpy().reshape(-1), classes=[0, 1, 2])
    score = np.swapaxes(score, 1, 2).reshape(-1, nclass)
    return average_precision_score(true, score)

def accuracy(pred_cls, true_cls, nclass=3):
    """
    compute per-node classification accuracy
    """
    accu = []
    for i in range(nclass):
        intersect = ((pred_cls == i) + (true_cls == i)).eq(2).sum().item()
        thiscls = (true_cls == i).sum().item()
        accu.append(intersect / thiscls)
    return np.array(accu)

# W_loss = torch.tensor(np.load('weight.npy'))

# def acc_loss(x,y,climatology,Std,Mean,W_loss):

#     x = x*Std+Mean-climatology
#     y = y*Std+Mean-climatology
#     acc_cli = 0
#     for var in range(x.shape[1]):
#         for level in range(x.shape[2]):
#             for lat_idx in range(W_loss.shape[0]):
#                 acc_cli += W_loss[lat_idx]*(((x[:,var,level,lat_idx])*(y[:,var,level,lat_idx])).mean())/((((x[:,var,level,lat_idx]**2).mean())*((y[:,var,level,lat_idx]**2).mean()))**0.5)
#     loss =  acc_cli/(x.shape[1]*x.shape[2])
#     return (1-loss)

def acc_loss(x,y,climatology,Max,Min,W_loss):
    x = x*(Max-Min)+Min-climatology
    y = y*(Max-Min)+Min-climatology
    acc_cli = 0
    for var in range(x.shape[1]):
        for level in range(x.shape[2]):
            for lat_idx in range(W_loss.shape[0]):
                acc_cli += W_loss[lat_idx]*(((x[:,var,level,lat_idx])*(y[:,var,level,lat_idx])).mean())/((((x[:,var,level,lat_idx]**2).mean())*((y[:,var,level,lat_idx]**2).mean()))**0.5)
    loss =  acc_cli/(x.shape[1]*x.shape[2])
    return torch.sqrt(1-loss)

# W_loss = torch.tensor(np.load('weight.npy'))

def weithted_loss(out,true,criterion,W_loss):

    loss_ = 0
    for lat_idx in range(W_loss.shape[0]):
        loss_ += W_loss[lat_idx]*criterion(out[...,lat_idx,:],true[...,lat_idx,:])
    return loss_

def train(args, model, train_loader, optimizer, epoch, device, logger):
    # if args.balance:
    #     w = torch.tensor(np.random.rand(3)).to(device)
    # else:
    #     w = torch.tensor([1.0,1.0,1.0]).to(device)
    model.train()
    tot_loss = 0
    count_ = 0
    count = 0
    # W_loss = torch.tensor(np.load('weight.npy')).to(device)
    for batch_idx,(data, label) in tqdm(enumerate(train_loader)):
        count_+=1
        target = label
        data, target = data.float().to(device), target.float().to(device) #前五个量是zqtuv
        optimizer.zero_grad()
        criterion = nn.MSELoss()
        output = model(data) #这里输入训练数据 bs x 16 x 10242 x 37

        loss = criterion(output, target) #weithted_loss(output, target, criterion, W_loss) #acc_loss(output, target) #MSE求loss

        loss.backward()#反向传播求梯度
        optimizer.step()#梯度下降
        tot_loss += loss.item()#给出最新loss
        count += data.size()[0]#计数君
        if batch_idx % args.log_interval == 0:
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    tot_loss /= count
    torch.cuda.empty_cache()
    return tot_loss



def test(args, model, test_loader, device, logger):
    # label frequencies: [0.001020786726132422, 0.9528737404907279, 0.04610547278313972]
    # if args.balance:
    #     w = torch.tensor([0.00102182, 0.95426438, 0.04471379]).to(device)
    # else:
    #     w = torch.tensor([1.0,1.0,1.0]).to(device)
    model.eval() #推理模式
    test_loss = 0
    ious = np.zeros(3)
    accus = np.zeros(3)
    aps = 0
    count_ = 0
    count = 0
    corr_ = 0
    target_set = []
    acc_set = []
    rmse_loss = 0
    criterion = nn.MSELoss()
    mean_list=np.load('mean_list.npy')
    std_list=np.load('std_list.npy')
    mean_list=torch.tensor(mean_list).to(device)
    std_list=torch.tensor(std_list).to(device)
    best_results=[]
    # W_loss = torch.tensor(np.load('weight.npy')).to(device)
    with torch.no_grad():
        for data, label in tqdm(test_loader):
            target = label 
            original_target = label.to(device)
            data, target = data.to(device).float(), target.to(device).float()
            output = model(data) #输入数据直接推理
            test_loss += criterion(output, target) #weithted_loss(output, target, criterion, W_loss).item() # sum up batch loss
            # test_loss += acc_loss(output,target,climatology_,Max,Min,W_loss)
            for idxx in range(output.shape[1]):
                original_target[:,idxx]=target[:,idxx]*std_list[idxx]+mean_list[idxx] #std_list[idxx]+mean_list[idxx]
                output[:,idxx]=output[:,idxx]*std_list[idxx]+mean_list[idxx]
                # print(std_list.device)
            rmse_loss += criterion(output[:,-2], original_target[:,-2]).item()
            # output_set.append(np.array(out_temp.cpu(),dtype=np.float32)) #必须是相同的精度才能算corr
            # print(test_loss)
            count_ += 1
            data11=data.to('cpu')
            for b_s in range(data11.shape[0]):
                temp_=torch.concat([data11[b_s], output[b_s].to('cpu')], dim=0)
                best_results.append(np.array(temp_))
                
            # corr_ += corr
    # test_loss=test_loss/count_
    # np.save('corr.npy',corr_/count_)
    # print('corr=',(corr_.mean())/count_)
    # CORR = (corr_.mean())/count_
    # Acc_z500 = np.mean(acc_set)
    # print('ACC of z500 =',Acc_z500)

    rmse_loss /= count_
    RMSE = np.sqrt(np.array(rmse_loss))

    # test_loss /= len(test_loader.dataset)
    #logger.info('Test set: Avg Precision: {:.4f}; MIoU: {:.4f}; Accu: {:.4f}, {:.4f}, {:.4f}; IoU: {:.4f}, {:.4f}, {:.4f}; Avg loss: {:.4f}'.format(
        #aps, np.mean(ious), accus[0], accus[1], accus[2], ious[0], ious[1], ious[2], test_loss))
    logger.info('Test set:{:.4f} Avg loss: {:.4f}'.format(RMSE,rmse_loss))
    return RMSE,best_results
    
def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--forecast_len', type=int, default=1, metavar='N',
                        help='input forecast length (default: 7)')
    parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=1e-2, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--mesh_folder', type=str, default="/home/gnn/data_process/meshcnn",
                        help='path to mesh folder (default: /home/gnn/data_process/meshcnn)')
    parser.add_argument('--data_folder', type=str, default="processed_data",
                        help='path to data folder (default: processed_data)')
    parser.add_argument('--climatology_dir', type=str, default="processed_data",
                        help='path to data folder (default: processed_data)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--max_level', type=int, default=6, help='max mesh level')
    parser.add_argument('--min_level', type=int, default=0, help='min mesh level')
    parser.add_argument('--feat', type=int, default=5, help='filter dimensions')
    parser.add_argument('--log_dir', type=str, default="log",
                        help='log directory for run')
    parser.add_argument('--decay', action="store_true", help="switch to decay learning rate")
    parser.add_argument('--optim', type=str, default="adam", choices=["adam", "sgd"])
    parser.add_argument('--resume', type=str, default=None, help="path to checkpoint if resume is needed")
    parser.add_argument('--balance', action="store_true", help="switch for label frequency balancing")
    parser.add_argument('--testmode', action="store_true", help="switch for test")

    args = parser.parse_args()

    # logger and snapshot current code
    # if not os.path.exists(args.log_dir):
    #     os.mkdir(args.log_dir)
    # shutil.copy2(__file__, os.path.join(args.log_dir, "script.py"))
    # shutil.copy2("model.py", os.path.join(args.log_dir, "model.py"))
    # shutil.copy2("run.sh", os.path.join(args.log_dir, "run.sh"))

    logger = logging.getLogger("train")
    logger.setLevel(logging.DEBUG)
    logger.handlers = []
    ch = logging.StreamHandler()
    logger.addHandler(ch)
    fh = logging.FileHandler(os.path.join(args.log_dir, "log.txt"))
    logger.addHandler(fh)
    logger.info("%s", repr(args))
    # nshape_edge = (721*1440,10*4**args.max_level+2)
    nshape_edge = (10*4**args.max_level+2,721*1440)

    torch.manual_seed(args.seed)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    trainset = ClimateSegLoader("train")
    valset = ClimateSegLoader("val")

    # train_sampler = ds(trainset, shuffle=True)
    # val_sampler = ds(valset, shuffle=False)

    # train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=False, drop_last=True, sampler=train_sampler)
    # val_loader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=False, drop_last=True, sampler=val_sampler)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=10)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=args.test_batch_size, shuffle=False, drop_last=True, num_workers=10)

    model = Multi_uunet(bs=args.batch_size, mesh_folder=args.mesh_folder, in_ch=2, out_ch=22, max_level=args.max_level, min_level=args.min_level, fdim=args.feat) #加入bs和fl，用bs代表输入天数（batch size），用fl代表预测时
    
    model = model.to(device)
    model = nn.DataParallel(model)

    if args.resume:
        resume_dict = torch.load(args.resume)

        def load_my_state_dict(self, state_dict, exclude='none'):
            from torch.nn.parameter import Parameter
    
            own_state = self.state_dict()
            for name, param in state_dict.items():
                if name not in own_state:
                    continue
                if exclude in name:
                    continue
                if isinstance(param, Parameter):
                    # backwards compatibility for serialized parameters
                    param = param.data
                own_state[name].copy_(param)

        load_my_state_dict(model, resume_dict)  

    logger.info("{} paramerters in total".format(sum(x.numel() for x in model.parameters())))

    if os.path.exists('weights/epoch_weight_epoch.pth'):
        model.load_state_dict(torch.load('weights/epoch_weight_epoch.pth'))

    rmse,best_results = test(args, model, val_loader, device, logger)
    # best_05c=[]
    # best_100w=[]
    # max_05c=0
    # max_100w=0
    # max_idx_05c=0
    # max_idx_100w=0
    
    # list_05c=[]
    # list_100w=[]
    # for idx in range(len(best_results)): #best_results:[[data,output],...]
    #     list_05c.append(best_results[idx][-2])
    #     list_100w.append(best_results[idx][-1])
    #     if max_05c<best_results[idx][1][-2]:
    #         max_05c=best_results[idx][1][-2]
    #         max_idx_05c=idx
    #     if max_100w<best_results[idx][1][-1]:
    #         max_100w=best_results[idx][1][-2]
    #         max_idx_100w=idx
    # best_05c.append(best_results[max_idx_05c])
    # best_100w.append(best_100w[max_idx_100w])
    # best_05c_index=find_top_10_indices_advanced(list_05c)
    # best_100w_index=find_top_10_indices_advanced(list_100w)
    np.save('output_test.npy',best_results)
    # print(best_results)
    # np.savetxt('best_05c_index.txt', best_results[best_05c_index], fmt='%d', delimiter=',')
    # np.savetxt('best_05c_index.txt', best_results[best_100w_index], fmt='%d', delimiter=',')
    print('模型结果的rmse=',rmse)


if __name__ == "__main__":
    main()