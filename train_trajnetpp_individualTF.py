import argparse
import baselineUtils
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import os
import time
from transformer.batch import subsequent_mask
from torch.optim import Adam,SGD, RMSprop, Adagrad
from transformer.noam_opt import NoamOpt
import numpy as np
import scipy.io
import json
import pickle
from torch.utils.tensorboard import SummaryWriter

## TrajNet++
import trajnetplusplustools
from data_load_utils import prepare_data


def main():
    parser=argparse.ArgumentParser(description='Train the individual Transformer model')
    parser.add_argument('--dataset_folder',type=str,default='datasets')
    parser.add_argument('--dataset_name',type=str,default='zara1')
    parser.add_argument('--obs',type=int,default=9)
    parser.add_argument('--preds',type=int,default=12)
    parser.add_argument('--emb_size',type=int,default=512)
    parser.add_argument('--heads',type=int, default=8)
    parser.add_argument('--layers',type=int,default=6)
    parser.add_argument('--dropout',type=float,default=0.1)
    parser.add_argument('--cpu',action='store_true')
    parser.add_argument('--val_size',type=int, default=0)
    parser.add_argument('--verbose',action='store_true')
    parser.add_argument('--max_epoch',type=int, default=1500)
    parser.add_argument('--batch_size',type=int,default=70)
    parser.add_argument('--validation_epoch_start', type=int, default=30)
    parser.add_argument('--resume_train',action='store_true')
    parser.add_argument('--delim',type=str,default='\t')
    parser.add_argument('--name', type=str, default="zara1")
    parser.add_argument('--factor', type=float, default=1.)
    parser.add_argument('--save_step', type=int, default=1)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--evaluate', type=bool, default=True)
    parser.add_argument('--model_pth', type=str)




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
        os.mkdir('output/Individual')
    except:
        pass
    try:
        os.mkdir(f'models/Individual')
    except:
        pass

    try:
        os.mkdir(f'output/Individual/{args.name}')
    except:
        pass

    try:
        os.mkdir(f'models/Individual/{args.name}')
    except:
        pass

    log=SummaryWriter('logs/Ind_%s'%model_name)

    log.add_scalar('eval/mad', 0, 0)
    log.add_scalar('eval/fad', 0, 0)
    device=torch.device("cuda")

    if args.cpu or not torch.cuda.is_available():
        device=torch.device("cpu")

    args.verbose=True


    ## TrajNet++ Edit ################################################################################################
    ## creation of the dataloaders for train and validation
    # if args.val_size==0:
    #     train_dataset,_ = baselineUtils.create_dataset(args.dataset_folder,args.dataset_name,0,args.obs,args.preds,delim=args.delim,train=True,verbose=args.verbose)
    #     val_dataset, _ = baselineUtils.create_dataset(args.dataset_folder, args.dataset_name, 0, args.obs,
    #                                                                 args.preds, delim=args.delim, train=False,
    #                                                                 verbose=args.verbose)
    # else:
    #     train_dataset, val_dataset = baselineUtils.create_dataset(args.dataset_folder, args.dataset_name, args.val_size,args.obs,
    #                                                           args.preds, delim=args.delim, train=True,
    #                                                           verbose=args.verbose)
    # test_dataset,_ =  baselineUtils.create_dataset(args.dataset_folder,args.dataset_name,0,args.obs,args.preds,delim=args.delim,train=False,eval=True,verbose=args.verbose)

    args.dataset_name = 'datasets/' + args.dataset_name
    ## Prepare data
    train_dataset, _, _ = prepare_data(args.dataset_name, subset='/train/', sample=1.0)
    val_dataset, _, _ = prepare_data(args.dataset_name, subset='/val/', sample=1.0)
    test_dataset, _, _ = prepare_data(args.dataset_name, subset='/val/', sample=1.0)
    ############################################################################################################

    import individual_TF
    model=individual_TF.IndividualTF(2, 3, 3, N=args.layers,
                   d_model=args.emb_size, d_ff=2048, h=args.heads, dropout=args.dropout,mean=[0,0],std=[0,0]).to(device)
    if args.resume_train:
        model.load_state_dict(torch.load(f'models/Individual/{args.name}/{args.model_pth}'))

    ## TrajNet++ ################################################################################################
    # tr_dl = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    # val_dl = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    # test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    ############################################################################################################

    #optim = SGD(list(a.parameters())+list(model.parameters())+list(generator.parameters()),lr=0.01)
    #sched=torch.optim.lr_scheduler.StepLR(optim,0.0005)
    optim = NoamOpt(args.emb_size, args.factor, (len(train_dataset) // args.batch_size)*args.warmup,
                        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    #optim=Adagrad(list(a.parameters())+list(model.parameters())+list(generator.parameters()),lr=0.01,lr_decay=0.001)
    epoch=0


    ## TrajNet++ ################################################################################################
    # mean=torch.cat((train_dataset[:]['src'][:,1:,2:4],train_dataset[:]['trg'][:,:,2:4]),1).mean((0,1))
    # std=torch.cat((train_dataset[:]['src'][:,1:,2:4],train_dataset[:]['trg'][:,:,2:4]),1).std((0,1))
    # means=[]
    # stds=[]
    # for i in np.unique(train_dataset[:]['dataset']):
    #     ind=train_dataset[:]['dataset']==i
    #     means.append(torch.cat((train_dataset[:]['src'][ind, 1:, 2:4], train_dataset[:]['trg'][ind, :, 2:4]), 1).mean((0, 1)))
    #     stds.append(
    #         torch.cat((train_dataset[:]['src'][ind, 1:, 2:4], train_dataset[:]['trg'][ind, :, 2:4]), 1).std((0, 1)))
    # mean=torch.stack(means).mean(0)
    # std=torch.stack(stds).mean(0)
    means=[]
    stds=[]
    for scene_i, (filename, scene_id, paths) in enumerate(train_dataset):
        ## make new scene
        scene = trajnetplusplustools.Reader.paths_to_xy(paths)
        scene = scene.transpose(1, 0, 2)[0:1]   # first ped only
        vel_scene = scene[:, 1:] - scene[:, :-1]
        vel_scene = torch.Tensor(vel_scene)
        means.append(vel_scene.mean((0, 1)))
        stds.append(vel_scene.std((0, 1)))
    mean=torch.stack(means).mean(0)
    std=torch.stack(stds).mean(0)
    ############################################################################################################

    scipy.io.savemat(f'models/Individual/{args.name}/norm.mat',{'mean':mean.cpu().numpy(),'std':std.cpu().numpy()})


    while epoch<args.max_epoch:
        epoch_loss=0
        model.train()

    ## TrajNet++ Edit ################################################################################################
        batch = {'src': [], 'trg': []}
        # for id_b,batch in enumerate(tr_dl):
        for id_b, (filename, scene_id, paths) in enumerate(train_dataset):
            ## make new scene
            pos_scene = trajnetplusplustools.Reader.paths_to_xy(paths)[:, 0]  # primary ped
            vel_scene = np.zeros_like(pos_scene)
            vel_scene[1:] = pos_scene[1:] - pos_scene[:-1]
            attr_scene = np.concatenate((pos_scene, vel_scene), axis=1)
            batch['src'].append(attr_scene[:args.obs])
            batch['trg'].append(attr_scene[-args.preds:])

            if len(batch['src']) % args.batch_size != 0:
                continue
            
            batch['src'] = torch.Tensor(np.stack(batch['src']))
            batch['trg'] = torch.Tensor(np.stack(batch['trg']))
    ############################################################################################################
            optim.optimizer.zero_grad()
            inp=(batch['src'][:,1:,2:4].to(device)-mean.to(device))/std.to(device)
            target=(batch['trg'][:,:-1,2:4].to(device)-mean.to(device))/std.to(device)
            target_c=torch.zeros((target.shape[0],target.shape[1],1)).to(device)
            target=torch.cat((target,target_c),-1)
            start_of_seq = torch.Tensor([0, 0, 1]).unsqueeze(0).unsqueeze(1).repeat(target.shape[0],1,1).to(device)

            dec_inp = torch.cat((start_of_seq, target), 1)

            src_att = torch.ones((inp.shape[0], 1,inp.shape[1])).to(device)
            trg_att=subsequent_mask(dec_inp.shape[1]).repeat(dec_inp.shape[0],1,1).to(device)




            pred=model(inp, dec_inp, src_att, trg_att)

            loss = F.pairwise_distance(pred[:, :,0:2].contiguous().view(-1, 2),
                                       ((batch['trg'][:, :, 2:4].to(device)-mean.to(device))/std.to(device)).contiguous().view(-1, 2).to(device)).mean() + torch.mean(torch.abs(pred[:,:,2]))
            loss.backward()
            optim.step()
            print("train epoch %03i/%03i  batch %04i / %04i loss: %7.4f" % (epoch, args.max_epoch, id_b, len(train_dataset), loss.item()))
            epoch_loss += loss.item()
    ## TrajNet++ Edit ################################################################################################
            batch = {'src': [], 'trg': []}
    ############################################################################################################
        #sched.step()
        log.add_scalar('Loss/train', epoch_loss * args.batch_size / len(train_dataset), epoch)



        with torch.no_grad():
            model.eval()

            val_loss=0
            step=0
            model.eval()
            gt = []
            pr = []
            inp_ = []
            peds = []
            frames = []
            dt = []
            dt_names = []

    ## TrajNet++ Edit ################################################################################################
            batch = {'src': [], 'trg': []}
            # for id_b, batch in enumerate(val_dl):
            for id_b, (filename, scene_id, paths) in enumerate(val_dataset):
                ## make new scene
                pos_scene = trajnetplusplustools.Reader.paths_to_xy(paths)[:, 0]  # primary ped
                vel_scene = np.zeros_like(pos_scene)
                vel_scene[1:] = pos_scene[1:] - pos_scene[:-1]
                attr_scene = np.concatenate((pos_scene, vel_scene), axis=1)
                batch['src'].append(attr_scene[:args.obs])
                batch['trg'].append(attr_scene[-args.preds:])

                if len(batch['src']) % args.batch_size != 0:
                    continue
                
                batch['src'] = torch.Tensor(np.stack(batch['src']))
                batch['trg'] = torch.Tensor(np.stack(batch['trg']))
    ############################################################################################################

                inp_.append(batch['src'])
                gt.append(batch['trg'][:, :, 0:2])
                # frames.append(batch['frames'])
                # peds.append(batch['peds'])
                # dt.append(batch['dataset'])

                inp = (batch['src'][:, 1:, 2:4].to(device) - mean.to(device)) / std.to(device)
                src_att = torch.ones((inp.shape[0], 1, inp.shape[1])).to(device)
                start_of_seq = torch.Tensor([0, 0, 1]).unsqueeze(0).unsqueeze(1).repeat(inp.shape[0], 1, 1).to(
                    device)
                dec_inp = start_of_seq

                for i in range(args.preds):
                    trg_att = subsequent_mask(dec_inp.shape[1]).repeat(dec_inp.shape[0], 1, 1).to(device)
                    out = model(inp, dec_inp, src_att, trg_att)
                    dec_inp = torch.cat((dec_inp, out[:, -1:, :]), 1)

                preds_tr_b = (dec_inp[:, 1:, 0:2] * std.to(device) + mean.to(device)).cpu().numpy().cumsum(1) + batch[
                                                                                                                    'src'][
                                                                                                                :, -1:,
                                                                                                                0:2].cpu().numpy()
                pr.append(preds_tr_b)
                print("val epoch %03i/%03i  batch %04i / %04i" % (
                    epoch, args.max_epoch, id_b, len(val_dataset)))
    ## TrajNet++ Edit ################################################################################################
                batch = {'src': [], 'trg': []}
    ############################################################################################################
  

            # peds = np.concatenate(peds, 0)
            # frames = np.concatenate(frames, 0)
            # dt = np.concatenate(dt, 0)
            gt = np.concatenate(gt, 0)
            # dt_names = test_dataset.data['dataset_name']
            pr = np.concatenate(pr, 0)
            mad, fad, errs = baselineUtils.distance_metrics(gt, pr)
            log.add_scalar('validation/MAD', mad, epoch)
            log.add_scalar('validation/FAD', fad, epoch)
            print("Val MAD FAD: ", mad, fad)


            if args.evaluate:

                model.eval()
                gt = []
                pr = []
                inp_ = []
                peds = []
                frames = []
                dt = []
    ## TrajNet++ Edit ################################################################################################
                batch = {'src': [], 'trg': []}
                # for id_b,batch in enumerate(test_dl):
                for id_b, (filename, scene_id, paths) in enumerate(test_dataset):
                    ## make new scene
                    pos_scene = trajnetplusplustools.Reader.paths_to_xy(paths)[:, 0]  # primary ped
                    vel_scene = np.zeros_like(pos_scene)
                    vel_scene[1:] = pos_scene[1:] - pos_scene[:-1]
                    attr_scene = np.concatenate((pos_scene, vel_scene), axis=1)
                    batch['src'].append(attr_scene[:args.obs])
                    batch['trg'].append(attr_scene[-args.preds:])

                    if len(batch['src']) % args.batch_size != 0:
                        continue
                    
                    batch['src'] = torch.Tensor(np.stack(batch['src']))
                    batch['trg'] = torch.Tensor(np.stack(batch['trg']))
    ############################################################################################################

                    inp_.append(batch['src'])
                    gt.append(batch['trg'][:,:,0:2])
                    # frames.append(batch['frames'])
                    # peds.append(batch['peds'])
                    # dt.append(batch['dataset'])

                    inp = (batch['src'][:, 1:, 2:4].to(device) - mean.to(device)) / std.to(device)
                    src_att = torch.ones((inp.shape[0], 1, inp.shape[1])).to(device)
                    start_of_seq = torch.Tensor([0, 0, 1]).unsqueeze(0).unsqueeze(1).repeat(inp.shape[0], 1, 1).to(
                        device)
                    dec_inp=start_of_seq

                    for i in range(args.preds):
                        trg_att = subsequent_mask(dec_inp.shape[1]).repeat(dec_inp.shape[0], 1, 1).to(device)
                        out = model(inp, dec_inp, src_att, trg_att)
                        dec_inp=torch.cat((dec_inp,out[:,-1:,:]),1)


                    preds_tr_b=(dec_inp[:,1:,0:2]*std.to(device)+mean.to(device)).cpu().numpy().cumsum(1)+batch['src'][:,-1:,0:2].cpu().numpy()
                    pr.append(preds_tr_b)
                    print("test epoch %03i/%03i  batch %04i / %04i" % (
                    epoch, args.max_epoch, id_b, len(test_dataset)))
    ## TrajNet++ Edit ################################################################################################
                    batch = {'src': [], 'trg': []}
    ############################################################################################################
  
                # peds = np.concatenate(peds, 0)
                # frames = np.concatenate(frames, 0)
                # dt = np.concatenate(dt, 0)
                gt = np.concatenate(gt, 0)
                # dt_names = test_dataset.data['dataset_name']
                pr = np.concatenate(pr, 0)
                mad, fad, errs = baselineUtils.distance_metrics(gt, pr)


                log.add_scalar('eval/DET_mad', mad, epoch)
                log.add_scalar('eval/DET_fad', fad, epoch)
                print("Test MAD FAD: ", mad, fad)

                # log.add_scalar('eval/DET_mad', mad, epoch)
                # log.add_scalar('eval/DET_fad', fad, epoch)

                scipy.io.savemat(f"output/Individual/{args.name}/det_{epoch}.mat",
                                 {'input': inp, 'gt': gt, 'pr': pr, 'peds': peds, 'frames': frames, 'dt': dt,
                                  'dt_names': dt_names})

        if epoch%args.save_step==0:

            torch.save(model.state_dict(),f'models/Individual/{args.name}/{epoch:05d}.pth')



        epoch+=1
    ab=1































if __name__=='__main__':
    main()
