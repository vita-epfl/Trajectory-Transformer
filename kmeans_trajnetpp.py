import argparse
import baselineUtils
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import os
import time

import numpy as np
import scipy.io
import json
import pickle
import kmeans_pytorch.kmeans as kmeans

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("agg")

## TrajNet++
import trajnetplusplustools
from data_load_utils import prepare_data


def main():
    parser=argparse.ArgumentParser(description='Train the individual Transformer model')
    parser.add_argument('--dataset_folder',type=str,default='datasets')
    parser.add_argument('--dataset_name',type=str,default='eth')
    parser.add_argument('--obs',type=int,default=8)
    parser.add_argument('--preds',type=int,default=12)
    parser.add_argument('--emb_size',type=int,default=1024)
    parser.add_argument('--heads',type=int, default=8)
    parser.add_argument('--layers',type=int,default=6)
    parser.add_argument('--dropout',type=float,default=0.1)
    parser.add_argument('--cpu',action='store_true')
    parser.add_argument('--output_folder',type=str,default='Output')
    parser.add_argument('--val_size',type=int, default=50)
    parser.add_argument('--verbose',action='store_true')
    parser.add_argument('--max_epoch',type=int, default=100)
    parser.add_argument('--batch_size',type=int,default=256)
    parser.add_argument('--validation_epoch_start', type=int, default=30)
    parser.add_argument('--resume_train',action='store_true')
    parser.add_argument('--delim',type=str,default='\t')
    parser.add_argument('--name', type=str, default="test_rot")
    parser.add_argument('--num_clusters', type=int, default=1000)
    parser.add_argument('--max_samples', type=int, default=200000)
    parser.add_argument('--scale', type=bool, default="True")
    parser.add_argument('--rot', type=bool, default="False")
    parser.add_argument('--axes_lim', type=int, default=2)
    parser.add_argument('--sample',type=float,default=1.0)






    args=parser.parse_args()
    model_name=args.name


    device=torch.device("cuda")

    if args.cpu or not torch.cuda.is_available():
        device=torch.device("cpu")

    args.verbose=True

    outdir=f'{args.dataset_name}_{args.num_clusters}_{args.max_samples}_scale{args.scale}_rot{args.rot}'
    try:
        os.mkdir(outdir)
    except:
        pass

    ## TrajNet++ ################################################################################################
    ## creation of the dataloaders for train and validation
    args.dataset_name = 'datasets/' + args.dataset_name
    ## Prepare data
    train_dataset, _, _ = prepare_data(args.dataset_name, subset='/train/', sample=args.sample)
    test_dataset, _, _ = prepare_data(args.dataset_name, subset='/test_private/', sample=args.sample)


    t = []
    for scene_i, (filename, scene_id, paths) in enumerate(train_dataset):
        ## make new scene
        scene = trajnetplusplustools.Reader.paths_to_xy(paths)
        scene = scene.transpose(1, 0, 2)[0:1]   # first ped only
        vel_scene = scene[:, 1:] - scene[:, :-1]
        vel_scene = torch.Tensor(vel_scene).reshape(-1, 2)
        t.append(vel_scene)
    t = torch.cat(t)

    test = []
    for scene_i, (filename, scene_id, paths) in enumerate(test_dataset):
        ## make new scene
        scene = trajnetplusplustools.Reader.paths_to_xy(paths)
        scene = scene.transpose(1, 0, 2)[0:1]   # first ped only
        vel_scene = scene[:, 1:] - scene[:, :-1]
        vel_scene = torch.Tensor(vel_scene).reshape(-1, 2)
        test.append(vel_scene)
    test = torch.cat(test)
    ############################################################################################################

    t= t.cpu().numpy()
    if args.scale:
        s=[1,0.7,0.5,1.5,2]
        t2=[]
        for i in s:
            t2.append(t*i)
        t=np.concatenate(t2,0)
    plt.figure()
    plt.scatter(t[:,0],t[:,1])
    plt.xlim([-args.axes_lim,args.axes_lim])
    plt.ylim([-args.axes_lim, args.axes_lim])
    plt.savefig(f'{outdir}/train_distribution.png')
    plt.close()

    test = test.cpu().numpy()


    plt.figure()
    plt.scatter(t[:,0],t[:,1],)
    plt.scatter(test[:,0],test[:,1])
    plt.xlim([-args.axes_lim,args.axes_lim])
    plt.ylim([-args.axes_lim, args.axes_lim])
    plt.savefig(f'{outdir}/test_distribution.png')
    plt.close()


    plt.figure()
    plt.scatter(test[:, 0], test[:, 1],c=['orange']*test.shape[0])
    plt.scatter(t[:, 0], t[:, 1],c=['blue']*t.shape[0], alpha=0.01)
    plt.xlim([-args.axes_lim, args.axes_lim])
    plt.ylim([-args.axes_lim, args.axes_lim])
    plt.savefig(f'{outdir}/test_distribution_alpha.png')
    plt.close()



    args.max_samples=min(args.max_samples,t.shape[0])
    ind = np.random.choice(t.shape[0], args.max_samples, replace=False)
    ended,cluster_index, center = kmeans.lloyd(t[ind], args.num_clusters,tol=1e-3)

    if ended==False:
        print("kmeans è crashato")
        return 0
    plt.figure()
    plt.set_cmap('prism')
    plt.scatter(t[ind, 0], t[ind, 1], c=cluster_index)
    plt.xlim([-args.axes_lim, args.axes_lim])
    plt.ylim([-args.axes_lim, args.axes_lim])
    plt.savefig(f'{outdir}/cluster_train_distribution_limited_data.png')
    plt.close()

    import scipy.spatial.distance as dist

    dist_tr = dist.cdist(t, center)
    dist_tr.min(axis=1).mean()

    dist_test = dist.cdist(test, center)
    dist_test.min(axis=1).mean()


    plt.figure()
    plt.set_cmap('prism')
    plt.scatter(t[:, 0], t[:, 1], c=dist_tr.argmin(1))
    plt.xlim([-args.axes_lim, args.axes_lim])
    plt.ylim([-args.axes_lim, args.axes_lim])
    plt.savefig(f'{outdir}/cluster_train_distribution_full_data.png')
    plt.close()



    plt.figure()
    plt.set_cmap('prism')

    plt.scatter(test[:, 0], test[:, 1], c=dist_test.argmin(1))
    plt.xlim([-args.axes_lim, args.axes_lim])
    plt.ylim([-args.axes_lim, args.axes_lim])
    plt.savefig(f'{outdir}/cluster_test_distribution_full_data.png')
    plt.close()

    plt.boxplot([dist_tr.min(axis=1),dist_test.min(axis=1)])
    plt.savefig(f'{outdir}/residuals.png')
    plt.close()

    cluster_count_tr= np.bincount(dist_tr.argmin(1))
    cluster_count_te= np.bincount(dist_test.argmin(1))

    plt.figure()
    plt.bar(np.arange(len(cluster_count_tr)),cluster_count_tr)
    plt.savefig(f'{outdir}/cluster_use_tr.png')
    plt.close()

    plt.figure()
    plt.bar(np.arange(len(cluster_count_te)),cluster_count_te)
    plt.savefig(f'{outdir}/cluster_use_te.png')
    plt.close()


    with open(f'{outdir}/stats.json','w') as f:
        j={"train":{
            "mean":dist_tr.min(axis=1).mean(),
            "std": dist_tr.min(axis=1).std(),
            "n_points":int(dist_tr.shape[0]),
            "avg_clust_presence":cluster_count_tr.mean(),
            "min_clust_presence": int(cluster_count_tr.min()),
            "max_clust_presence": int(cluster_count_tr.max()),
            "counts": cluster_count_tr.tolist()
        },

           "test":{
               "mean": dist_test.min(axis=1).mean(),
               "std": dist_test.min(axis=1).std(),
               "n_points":int(dist_test.shape[0]),
               "avg_clust_presence": cluster_count_te.mean(),
               "min_clust_presence": int(cluster_count_te.min()),
               "max_clust_presence": int(cluster_count_te.max()),
               "counts": cluster_count_te.tolist()
           }

        }

        json.dump(j,f)
    scipy.io.savemat(f'{outdir}/clusters.mat',{'centroids':center,"counts":cluster_count_tr})








if __name__=="__main__":
    main()





