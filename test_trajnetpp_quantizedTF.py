import argparse
import baselineUtils
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import os
import time
from transformer.batch import subsequent_mask
from torch.optim import Adam,SGD,RMSprop,Adagrad
from transformer.noam_opt import NoamOpt
import numpy as np
import scipy.io
import json
import pickle

from torch.utils.tensorboard import SummaryWriter
import quantized_TF

## TrajNet++
import trajnetplusplustools
from data_load_utils import prepare_data
from evaluator.eval_utils import trajnet_batch_eval, trajnet_batch_multi_eval
from trajnet_loader import trajnet_test_loader



def main():
    parser=argparse.ArgumentParser(description='Train the individual Transformer model')
    parser.add_argument('--dataset_folder',type=str,default='datasets')
    parser.add_argument('--dataset_name',type=str,default='zara1')
    parser.add_argument('--obs',type=int,default=8)
    parser.add_argument('--preds',type=int,default=12)
    parser.add_argument('--emb_size',type=int,default=512)
    parser.add_argument('--heads',type=int, default=8)
    parser.add_argument('--layers',type=int,default=6)
    parser.add_argument('--cpu',action='store_true')
    parser.add_argument('--verbose',action='store_true')
    parser.add_argument('--batch_size',type=int,default=256)
    parser.add_argument('--delim',type=str,default='\t')
    parser.add_argument('--name', type=str, default="zara1")
    parser.add_argument('--epoch',type=str,default="00001")
    parser.add_argument('--num_samples', type=int, default="20")




    args=parser.parse_args()
    model_name=args.name

    try:
        os.mkdir('models')
    except:
        pass
    try:
        os.mkdir('output')
    except:
        pass
    try:
        os.mkdir('output/QuantizedTF')
    except:
        pass
    try:
        os.mkdir(f'models/QuantizedTF')
    except:
        pass

    try:
        os.mkdir(f'output/QuantizedTF/{args.name}')
    except:
        pass

    try:
        os.mkdir(f'models/QuantizedTF/{args.name}')
    except:
        pass


    device=torch.device("cuda")

    if args.cpu or not torch.cuda.is_available():
        device=torch.device("cpu")

    args.verbose=True


    test_dataset, _, _ = prepare_data('datasets/' + args.dataset_name, subset='/test_private/', sample=1.0)

    mat = scipy.io.loadmat(os.path.join(args.dataset_folder, args.dataset_name, "clusters.mat"))

    clusters=mat['centroids']

    model=quantized_TF.QuantizedTF(clusters.shape[0], clusters.shape[0]+1, clusters.shape[0], N=args.layers,
                   d_model=args.emb_size, d_ff=1024, h=args.heads).to(device)

    model.load_state_dict(torch.load(f'models/QuantizedTF/{args.name}/{args.epoch}.pth'))
    model.to(device)

    # DETERMINISTIC MODE
    with torch.no_grad():
        model.eval()
        gt=[]
        pr=[]
        inp_=[]
        peds=[]
        frames=[]
        dt=[]
        dt_names=[]
        batch = {'src': [], 'trg': []}

        # TrajNet Eval
        ade, fde, pred_col, gt_col = 0, 0, 0, 0 
        for id_b,(batch, batch_split) in enumerate(trajnet_test_loader(test_dataset, args)):   # TrajNet_loader
            scale = np.random.uniform(0.5, 2)
            # rot_mat = np.array([[np.cos(r), np.sin(r)], [-np.sin(r), np.cos(r)]])
            n_in_batch = batch['src'].shape[0]
            speeds_inp = batch['src'][:, 1:, 2:4]
            gt_b = batch['trg'][:, :, 0:2]
            inp = torch.tensor(
                scipy.spatial.distance.cdist(speeds_inp.reshape(-1, 2), clusters).argmin(axis=1).reshape(n_in_batch,
                                                                                                         -1)).to(
                device)
            src_att = torch.ones((inp.shape[0], 1,inp.shape[1])).to(device)
            start_of_seq = torch.tensor([clusters.shape[0]]).repeat(n_in_batch).unsqueeze(1).to(device)
            dec_inp = start_of_seq

            for i in range(args.preds):
                trg_att = subsequent_mask(dec_inp.shape[1]).repeat(n_in_batch, 1, 1).to(device)
                out = model(inp, dec_inp, src_att, trg_att)
                dec_inp=torch.cat((dec_inp,out[:,-1:].argmax(dim=2)),1)


            preds_tr_b=clusters[dec_inp[:,1:].cpu().numpy()].cumsum(1)+batch['src'][:,-1:,0:2].cpu().numpy()
            ade_b, fde_b, pred_col_b, gt_col_b = trajnet_batch_eval(preds_tr_b, gt_b.cpu().numpy(), batch_split.cpu().numpy())
            ade += ade_b
            fde += fde_b
            pred_col += pred_col_b
            gt_col += gt_col_b

        scipy.io.savemat(f"output/QuantizedTF/{args.name}/MM_deterministic.mat",{'input':inp,'gt':gt,'pr':pr,'peds':peds,'frames':frames,'dt':dt,'dt_names':dt_names})

        print("Determinitic:")
        print("mad: %6.3f"% (ade / len(test_dataset)))
        print("fad: %6.3f" % (fde / len(test_dataset)))
        print("pred_col: %6.3f"%(pred_col / len(test_dataset)))
        print("gt_col: %6.3f" % (gt_col/ len(test_dataset)))

        # MULTI MODALITY
        num_samples=args.num_samples

        model.eval()
        gt=[]
        pr_all={}
        inp_=[]
        peds=[]
        frames=[]
        dt=[]
        dt_names=[]

        topk_ade = 0.0
        topk_fde = 0.0
        for sam in range(num_samples):
            pr_all[sam]=[]
        for id_b,(batch, batch_split) in enumerate(trajnet_test_loader(test_dataset, args)):   # TrajNet_loader
            scale = np.random.uniform(0.5, 2)
            # rot_mat = np.array([[np.cos(r), np.sin(r)], [-np.sin(r), np.cos(r)]])
            n_in_batch = batch['src'].shape[0]
            speeds_inp = batch['src'][:, 1:, 2:4]
            gt_b = batch['trg'][:, :, 0:2]
            gt.append(gt_b)
            inp__=batch['src'][:,:,0:2]
            inp_.append(inp__)
            inp = torch.tensor(
                scipy.spatial.distance.cdist(speeds_inp.reshape(-1, 2), clusters).argmin(axis=1).reshape(n_in_batch,
                                                                                                         -1)).to(
                device)
            src_att = torch.ones((inp.shape[0], 1,inp.shape[1])).to(device)
            start_of_seq = torch.tensor([clusters.shape[0]]).repeat(n_in_batch).unsqueeze(1).to(device)

            multi_preds_b = []
            for sam in range(num_samples):
                dec_inp = start_of_seq

                for i in range(args.preds):
                    trg_att = subsequent_mask(dec_inp.shape[1]).repeat(n_in_batch, 1, 1).to(device)
                    out = model.predict(inp, dec_inp, src_att, trg_att)
                    h=out[:,-1]
                    dec_inp=torch.cat((dec_inp,torch.multinomial(h,1)),1)


                preds_tr_b=clusters[dec_inp[:,1:].cpu().numpy()].cumsum(1)+batch['src'][:,-1:,0:2].cpu().numpy()

                pr_all[sam].append(preds_tr_b)
                multi_preds_b.append(preds_tr_b)
            
            topk_ade_b, topk_fde_b = trajnet_batch_multi_eval(multi_preds_b, gt_b.cpu().numpy(), batch_split.cpu().numpy())
            topk_ade += topk_ade_b
            topk_fde += topk_fde_b

        print("Multimodality:")
        print("MM ADE: %6.3f" % (topk_ade / len(test_dataset)))
        print("MM FDE: %6.3f" % (topk_fde / len(test_dataset)))
































if __name__=='__main__':
    main()
