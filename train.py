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
#解除显存使用限制
use_cuda = torch.cuda.is_available() #not args.no_cuda and torch.cuda.is_available()
def setup_DDP(backend="nccl", verbose=False):
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    # If the OS is Windows or macOS, use gloo instead of nccl
    distri.init_process_group(backend=backend)
    # set distributed device
    device = torch.device("cuda:{}".format(local_rank))
    if verbose:
        print(f"local rank: {local_rank}, global rank: {rank}, world size: {world_size}")
    return rank, local_rank, world_size, device
# print("Setup_DDP step")
# rank, local_rank, world_size, device = setup_DDP(verbose=True)
# print("\n my rank is",rank)
# this_rank = distri.get_rank()
device = torch.device("cuda" if use_cuda else "cpu")

# msmm = np.load('mean_std.npz')
# msmm = np.load('mean_std_max_min.npz')
# Std = torch.tensor(msmm['std'][None,...,None].astype(np.float32)).to(device)
# Mean = torch.tensor(msmm['mean'][None,...,None,None].astype(np.float32)).to(device)
# Max = torch.tensor(msmm['max'][None,...,None,None].astype(np.float32)).to(device)
# Min = torch.tensor(msmm['min'][None,...,None,None].astype(np.float32)).to(device)
# Std = Std[:,:,level_2]
# Mean = Mean[:,:,level_2]
# Max = Max[:,:,level_2]
# Min = Min[:,:,level_2]
# mean_=msmm['mean'].astype(np.float32)
# std_=msmm['std'].astype(np.float32)
# max_=msmm['max'].astype(np.float32)
# min_=msmm['min'].astype(np.float32)

tgt_device = device
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
    # torch.cuda.empty_cache()
    return tot_loss

def test(args, model, test_loader, device, logger):
    # label frequencies: [0.001020786726132422, 0.9528737404907279, 0.04610547278313972]
    # if args.balance:
    #     w = torch.tensor([0.00102182, 0.95426438, 0.04471379]).to(device)
    # else:
    #     w = torch.tensor([1.0,1.0,1.0]).to(device)
    model.eval() #推理模式
    test_loss = 0
    count_ = 0
    mean_list=np.load('/home/xzw/retention_tian/data_preprocess/mean_list_inter.npy')
    std_list=np.load('/home/xzw/retention_tian/data_preprocess/std_list_inter.npy')
    rmse_loss = 0
    criterion = nn.MSELoss()
    temp_data=[]
    # W_loss = torch.tensor(np.load('weight.npy')).to(device)
    with torch.no_grad():
        for data, label in tqdm(test_loader):
            target = label #181*360
            # climatology_ = label[:,5:].to(device)
            #data = pre_load(data,edge_src_target, nshape_edge, max_level)
            # target_copy = target.clone().detach().cpu()
            # target_set.append(target_copy)
            # data = torch.tensor(data)
            data, target = data.to(device).float(), target.to(device).float()
            output = model(data) #输入数据直接推理
            
            test_loss += criterion(output, target) #weithted_loss(output, target, criterion, W_loss).item() # sum up batch loss
            for idxx in range(output.shape[1]):
                target[:,idxx]=target[:,idxx]*std_list[idxx]+mean_list[idxx] #std_list[idxx]+mean_list[idxx]
                output[:,idxx]=output[:,idxx]*std_list[idxx]+mean_list[idxx]
                # print(std_list.device)
            temp_data.append(np.array(torch.cat([data.cpu(),output.cpu()],dim=1)))
            rmse_loss += criterion(output[:,-1], target[:,-1]).item()

            # output_set.append(np.array(out_temp.cpu(),dtype=np.float32)) #必须是相同的精度才能算corr

            count_ += 1
            # corr_ += corr

    # np.save('corr.npy',corr_/count_)
    # print('corr=',(corr_.mean())/count_)
    # CORR = (corr_.mean())/count_
    # Acc_z500 = np.mean(acc_set)

    # print('ACC of z500 =',Acc_z500)
    np.save('result_temp.npy',temp_data)
    if count_==0:
        count_=1
    rmse_loss /= count_
    RMSE = np.sqrt(np.array(rmse_loss))

    # test_loss /= len(test_loader.dataset)
    #logger.info('Test set: Avg Precision: {:.4f}; MIoU: {:.4f}; Accu: {:.4f}, {:.4f}, {:.4f}; IoU: {:.4f}, {:.4f}, {:.4f}; Avg loss: {:.4f}'.format(
        #aps, np.mean(ious), accus[0], accus[1], accus[2], ious[0], ious[1], ious[2], test_loss))
    logger.info('Test set:{:.4f} Avg loss: {:.4f}'.format(RMSE,rmse_loss))
    return RMSE,rmse_loss
    
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
    parser.add_argument('--mesh_folder', type=str, default="/WdHeDisk/users/xuzhewen/data_process/meshcnn",
                        help='path to mesh folder (default: /WdHeDisk/users/xuzhewen/data_process/meshcnn)')
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

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=10)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=False, drop_last=True, num_workers=10)

    model = Multi_uunet(bs=args.batch_size, mesh_folder=args.mesh_folder, in_ch=22, out_ch=2, max_level=args.max_level, min_level=args.min_level, fdim=args.feat) #加入bs和fl，用bs代表输入天数（batch size），用fl代表预测时
    
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

    if args.optim == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    if args.decay:
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(30), eta_min=1e-4)

    best_ap = 0.1

    readpth=0
    checkpoint_path = os.path.join(args.log_dir, 'checkpoint_latest')
    for epoch in range(1, args.epochs + 1):
        if readpth==1:
            print('loading checkpoint')
            model.load_state_dict(torch.load('./weights/epoch_weight_epoch.pth'))
            # checkpoint = torch.load(checkpoint_path+'_UNet_best.pth.tar')
            # model.load_state_dict(checkpoint['state_dict'])
            # optimizer.load_state_dict(checkpoint['optimizer'])
            # epoch = checkpoint['epoch']
            # model.load_state_dict(torch.load('./weights/epoch_weight.pth'))
            readpth = 0
        if not args.testmode:
            loss = train(args, model, train_loader, optimizer, epoch, device, logger)
            if args.decay:
                scheduler.step()
        # state_dict = torch.load('temp_para.pth')
        # model.load_state_dict(state_dict['model'])
        rmse,rmse_loss = test(args, model, val_loader, device, logger)
        torch.save(model.state_dict(),'./weights/epoch_weight_epoch.pth')
        print('容量rmse为',rmse_loss)
        if rmse_loss < best_ap:
            best_ap = rmse
            is_best = True
        else:
            is_best = False
        # remove sparse matrices since they cannot be stored
        state_dict_no_sparse = [it for it in model.state_dict().items() if it[1].type() != "torch.cuda.sparse.FloatTensor"]
        state_dict_no_sparse = OrderedDict(state_dict_no_sparse)
        if is_best: # early stop
            break
        if not args.testmode:
            save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(), #state_dict_no_sparse,
            'best_ap': best_ap,
            'optimizer': optimizer.state_dict(),
            }, is_best, epoch, checkpoint_path, "_UNet", logger)

if __name__ == "__main__":
    main()
