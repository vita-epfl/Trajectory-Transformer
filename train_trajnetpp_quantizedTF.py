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

## TrajNet++
import trajnetplusplustools
from data_load_utils import prepare_data
from trajnet_loader import trajnet_loader



def main():
    parser=argparse.ArgumentParser(description='Train the individual Transformer model')
    parser.add_argument('--dataset_folder',type=str,default='datasets')
    parser.add_argument('--dataset_name',type=str,default='zara1')
    parser.add_argument('--obs',type=int,default=8)
    parser.add_argument('--preds',type=int,default=12)
    parser.add_argument('--emb_size',type=int,default=512)
    parser.add_argument('--heads',type=int, default=8)
    parser.add_argument('--layers',type=int,default=6)
    parser.add_argument('--dropout',type=float,default=0.1)
    parser.add_argument('--cpu',action='store_true')
    parser.add_argument('--output_folder',type=str,default='Output')
    parser.add_argument('--val_size',type=int, default=0)
    parser.add_argument('--gpu_device',type=str, default="0")
    parser.add_argument('--verbose',action='store_true')
    parser.add_argument('--max_epoch',type=int, default=100)
    parser.add_argument('--batch_size',type=int,default=100)
    parser.add_argument('--validation_epoch_start', type=int, default=30)
    parser.add_argument('--resume_train',action='store_true')
    parser.add_argument('--delim',type=str,default='\t')
    parser.add_argument('--name', type=str, default="zara1")
    parser.add_argument('--factor', type=float, default=1.)
    parser.add_argument('--evaluate',type=bool,default=True)
    parser.add_argument('--save_step', type=int, default=5)



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

    log=SummaryWriter('logs/%s'%model_name)

    #os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_device
    device=torch.device("cuda")

    if args.cpu or not torch.cuda.is_available():
        device=torch.device("cpu")

    args.verbose=True


    ## TrajNet++ ################################################################################################
    ## creation of the dataloaders for train and validation
    ## Prepare data
    train_dataset, _, _ = prepare_data('datasets/' + args.dataset_name, subset='/train/', sample=1.0)
    val_dataset, _, _ = prepare_data('datasets/' + args.dataset_name, subset='/val/', sample=1.0)
    test_dataset, _, _ = prepare_data('datasets/' + args.dataset_name, subset='/test_private/', sample=1.0)
    ############################################################################################################

    mat = scipy.io.loadmat(os.path.join(args.dataset_folder, args.dataset_name, "clusters.mat"))
    clusters=mat['centroids']

    import quantized_TF
    model=quantized_TF.QuantizedTF(clusters.shape[0], clusters.shape[0]+1, clusters.shape[0], N=args.layers,
                   d_model=args.emb_size, d_ff=1024, h=args.heads, dropout=args.dropout).to(device)


    ## TrajNet++ ################################################################################################
    optim = NoamOpt(args.emb_size, args.factor, (len(train_dataset) // args.batch_size)*5,
                        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    epoch=0



    while epoch<args.max_epoch:
        epoch_loss=0
        model.train()

        for id_b,batch in enumerate(trajnet_loader(train_dataset, args)):  # Trajnet Loader
            optim.optimizer.zero_grad()
            scale=np.random.uniform(0.5,4)
            #rot_mat = np.array([[np.cos(r), np.sin(r)], [-np.sin(r), np.cos(r)]])
            n_in_batch=batch['src'].shape[0]
            speeds_inp=batch['src'][:,1:,2:4]*scale
            inp=torch.tensor(scipy.spatial.distance.cdist(speeds_inp.reshape(-1,2),clusters).argmin(axis=1).reshape(n_in_batch,-1)).to(device)
            
            speeds_trg = batch['trg'][:,:,2:4]*scale
            target = torch.tensor(
                scipy.spatial.distance.cdist(speeds_trg.reshape(-1, 2), clusters).argmin(axis=1).reshape(n_in_batch, -1)).to(
                device)
            src_att = torch.ones((inp.shape[0], 1,inp.shape[1])).to(device)
            trg_att=subsequent_mask(target.shape[1]).repeat(n_in_batch,1,1).to(device)
            start_of_seq=torch.tensor([clusters.shape[0]]).repeat(n_in_batch).unsqueeze(1).to(device)
            dec_inp=torch.cat((start_of_seq,target[:,:-1]),1)



            out=model(inp, dec_inp, src_att, trg_att)

            loss = F.cross_entropy(out.view(-1,out.shape[-1]),target.view(-1),reduction='mean')
            loss.backward()
            optim.step()
            print("epoch %03i/%03i  frame %04i / %04i loss: %7.4f" % (epoch, args.max_epoch, id_b * args.batch_size, len(train_dataset), loss.item()))
            epoch_loss += loss.item()

        #sched.step()
        log.add_scalar('Loss/train', epoch_loss * args.batch_size / len(train_dataset), epoch)
        with torch.no_grad():
            model.eval()

            gt=[]
            pr=[]
            val_loss=0
            step=0
            for id_b, batch in enumerate(trajnet_loader(val_dataset, args)):  # Trajnet Loader
                # rot_mat = np.array([[np.cos(r), np.sin(r)], [-np.sin(r), np.cos(r)]])
                n_in_batch = batch['src'].shape[0]
                speeds_inp = batch['src'][:, 1:, 2:4]
                inp = torch.tensor(
                    scipy.spatial.distance.cdist(speeds_inp.contiguous().reshape(-1, 2), clusters).argmin(axis=1).reshape(n_in_batch,
                                                                                                             -1)).to(
                    device)
                speeds_trg = batch['trg'][:, :, 2:4]
                target = torch.tensor(
                    scipy.spatial.distance.cdist(speeds_trg.contiguous().reshape(-1, 2), clusters).argmin(axis=1).reshape(n_in_batch,
                                                                                                             -1)).to(
                    device)
                src_att = torch.ones((inp.shape[0], 1,inp.shape[1])).to(device)
                trg_att = subsequent_mask(target.shape[1]).repeat(n_in_batch, 1, 1).to(device)
                start_of_seq = torch.tensor([clusters.shape[0]]).repeat(n_in_batch).unsqueeze(1).to(device)
                dec_inp = torch.cat((start_of_seq, target[:, :-1]), 1)

                out = model(inp, dec_inp, src_att, trg_att)

                loss = F.cross_entropy(out.contiguous().view(-1, out.shape[-1]), target.contiguous().view(-1), reduction='mean')

                print("val epoch %03i/%03i  frame %04i / %04i loss: %7.4f" % (
                epoch, args.max_epoch, step * args.batch_size, len(val_dataset), loss.item()))
                val_loss+=loss.item()
                step+=1

            log.add_scalar('validation/loss', val_loss * args.batch_size / len(val_dataset), epoch)

            if args.evaluate:
                # DETERMINISTIC MODE
                model.eval()
                model.eval()
                gt = []
                pr = []
                inp_ = []
                peds = []
                frames = []
                dt = []
                dt_names = []
                for id_b,batch in enumerate(trajnet_loader(test_dataset, args)):  # Trajnet Loader
                    inp_.append(batch['src'][:,:,0:2])
                    gt.append(batch['trg'][:, :, 0:2])

                    n_in_batch = batch['src'].shape[0]
                    speeds_inp = batch['src'][:, 1:, 2:4]
                    gt_b = batch['trg'][:, :, 0:2]
                    inp = torch.tensor(
                        scipy.spatial.distance.cdist(speeds_inp.reshape(-1, 2), clusters).argmin(axis=1).reshape(n_in_batch,
                                                                                                                 -1)).to(
                        device)
                    src_att = torch.ones((inp.shape[0], 1,inp.shape[1])).to(device)
                    trg_att = subsequent_mask(target.shape[1]).repeat(n_in_batch, 1, 1).to(device)
                    start_of_seq = torch.tensor([clusters.shape[0]]).repeat(n_in_batch).unsqueeze(1).to(device)
                    dec_inp = start_of_seq

                    for i in range(args.preds):
                        trg_att = subsequent_mask(dec_inp.shape[1]).repeat(n_in_batch, 1, 1).to(device)
                        out = model(inp, dec_inp, src_att, trg_att)
                        dec_inp=torch.cat((dec_inp,out[:,-1:].argmax(dim=2)),1)


                    preds_tr_b=clusters[dec_inp[:,1:].cpu().numpy()].cumsum(1)+batch['src'][:,-1:,0:2].cpu().numpy()
                    pr.append(preds_tr_b)

                gt = np.concatenate(gt, 0)
                # dt_names = test_dataset.data['dataset_name']
                pr = np.concatenate(pr, 0)
                mad,fad,errs=baselineUtils.distance_metrics(gt,pr)
                print("Test MAD FAD: ", mad, fad)

                log.add_scalar('eval/DET_mad', mad, epoch)
                log.add_scalar('eval/DET_fad', fad, epoch)

                scipy.io.savemat(f"output/QuantizedTF/{args.name}/{epoch:05d}.mat",
                                 {'input': inp, 'gt': gt, 'pr': pr, 'peds': peds, 'frames': frames, 'dt': dt,
                                  'dt_names': dt_names})


                # MULTI MODALITY
                if True:
                    num_samples=3

                    model.eval()
                    gt=[]
                    pr_all={}
                    batch = {'src': [], 'trg': []}
                    for sam in range(num_samples):
                        pr_all[sam]=[]
                    for id_b,batch in enumerate(trajnet_loader(test_dataset, args)):  # Trajnet Loader
                        # rot_mat = np.array([[np.cos(r), np.sin(r)], [-np.sin(r), np.cos(r)]])
                        n_in_batch = batch['src'].shape[0]
                        speeds_inp = batch['src'][:, 1:, 2:4]
                        gt_b = batch['trg'][:, :, 0:2]
                        gt.append(gt_b)
                        inp = torch.tensor(
                            scipy.spatial.distance.cdist(speeds_inp.reshape(-1, 2), clusters).argmin(axis=1).reshape(n_in_batch,
                                                                                                                     -1)).to(
                            device)
                        src_att = torch.ones((inp.shape[0], 1,inp.shape[1])).to(device)
                        trg_att = subsequent_mask(target.shape[1]).repeat(n_in_batch, 1, 1).to(device)
                        start_of_seq = torch.tensor([clusters.shape[0]]).repeat(n_in_batch).unsqueeze(1).to(device)

                        for sam in range(num_samples):
                            dec_inp = start_of_seq

                            for i in range(args.preds):
                                trg_att = subsequent_mask(dec_inp.shape[1]).repeat(n_in_batch, 1, 1).to(device)
                                out = model.predict(inp, dec_inp, src_att, trg_att)
                                h=out[:,-1]
                                dec_inp=torch.cat((dec_inp,torch.multinomial(h,1)),1)


                            preds_tr_b=clusters[dec_inp[:,1:].cpu().numpy()].cumsum(1)+batch['src'][:,-1:,0:2].cpu().numpy()

                            pr_all[sam].append(preds_tr_b)

                    gt=np.concatenate(gt,0)
                    #pr=np.concatenate(pr,0)
                    samp = {}
                    for k in pr_all.keys():
                        samp[k] = {}
                        samp[k]['pr'] = np.concatenate(pr_all[k], 0)
                        samp[k]['mad'], samp[k]['fad'], samp[k]['err'] = baselineUtils.distance_metrics(gt, samp[k]['pr'])

                    ev = [samp[i]['err'] for i in range(num_samples)]
                    e20 = np.stack(ev, -1)
                    mad_samp=e20.mean(1).min(-1).mean()
                    fad_samp=e20[:,-1].min(-1).mean()
                    #mad,fad,errs=baselineUtils.distance_metrics(gt,pr)
                    
                    print("Test MM_MAD MM_FAD: ", mad_samp, fad_samp)

                    log.add_scalar('eval/MM_mad', mad_samp, epoch)
                    log.add_scalar('eval/MM_fad', fad_samp, epoch)

            if epoch % args.save_step == 0 and epoch > 80:
                torch.save(model.state_dict(), f'models/QuantizedTF/{args.name}/{epoch:05d}.pth')




        epoch+=1

    ab=1































if __name__=='__main__':
    main()
