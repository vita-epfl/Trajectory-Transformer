import os
import argparse
import pickle

from joblib import Parallel, delayed
import scipy
import torch
from tqdm import tqdm
import trajnetplusplustools
import numpy as np

from evaluator.trajnet_evaluator import trajnet_evaluate
from evaluator.write_utils import load_test_datasets, preprocess_test, write_predictions

# Transformer
import scipy.io
import quantized_TF
from transformer.batch import subsequent_mask


def predict_scene(model, clusters, paths, args):
    ## For each scene, get predictions
    ## Taken snippet from test_trajnetpp_quantizedTF.py
    batch = {'src': []}
    device = 'cuda:0'
    paths = preprocess_test(paths, args.obs_length)
    pos_scene = trajnetplusplustools.Reader.paths_to_xy(paths)  # ALL ped
    vel_scene = np.zeros_like(pos_scene)
    vel_scene[1:] = pos_scene[1:] - pos_scene[:-1]
    attr_scene = np.concatenate((pos_scene, vel_scene), axis=2)
    batch['src'] = torch.Tensor(attr_scene[:args.obs_length]).permute(1, 0, 2)

    scale = np.random.uniform(0.5, 2)
    n_in_batch = batch['src'].shape[0]
    speeds_inp = batch['src'][:, 1:, 2:4]
    inp = torch.tensor(
        scipy.spatial.distance.cdist(speeds_inp.reshape(-1, 2), clusters).argmin(axis=1).reshape(n_in_batch,
                                                                                                    -1)).to(
        device)
    src_att = torch.ones((inp.shape[0], 1,inp.shape[1])).to(device)
    start_of_seq = torch.tensor([clusters.shape[0]]).repeat(n_in_batch).unsqueeze(1).to(device)
    dec_inp = start_of_seq

    for i in range(args.pred_length):
        trg_att = subsequent_mask(dec_inp.shape[1]).repeat(n_in_batch, 1, 1).to(device)
        out = model(inp, dec_inp, src_att, trg_att)
        dec_inp=torch.cat((dec_inp,out[:,-1:].argmax(dim=2)),1)

    preds_tr_b=clusters[dec_inp[:,1:].cpu().numpy()].cumsum(1)+batch['src'][:,-1:,0:2].cpu().numpy()

    multimodal_outputs = {}
    multimodal_outputs[0] = [preds_tr_b[0], preds_tr_b[1:].transpose(1, 0, 2)]
    return multimodal_outputs


def predict_multimodal_scene(model, clusters, paths, args):
    ## For each scene, get predictions
    ## Taken snippet from test_trajnetpp_quantizedTF.py
    batch = {'src': []}
    device = 'cuda:0'
    paths = preprocess_test(paths, args.obs_length)
    pos_scene = trajnetplusplustools.Reader.paths_to_xy(paths)  # ALL ped
    vel_scene = np.zeros_like(pos_scene)
    vel_scene[1:] = pos_scene[1:] - pos_scene[:-1]
    attr_scene = np.concatenate((pos_scene, vel_scene), axis=2)
    batch['src'] = torch.Tensor(attr_scene[:args.obs_length]).permute(1, 0, 2)

    scale = np.random.uniform(0.5, 2)
    n_in_batch = batch['src'].shape[0]
    speeds_inp = batch['src'][:, 1:, 2:4]
    inp = torch.tensor(
        scipy.spatial.distance.cdist(speeds_inp.reshape(-1, 2), clusters).argmin(axis=1).reshape(n_in_batch,
                                                                                                    -1)).to(
        device)
    src_att = torch.ones((inp.shape[0], 1,inp.shape[1])).to(device)
    start_of_seq = torch.tensor([clusters.shape[0]]).repeat(n_in_batch).unsqueeze(1).to(device)

    multimodal_outputs = {}
    for sam in range(args.modes):
        dec_inp = start_of_seq

        for i in range(args.pred_length):
            trg_att = subsequent_mask(dec_inp.shape[1]).repeat(n_in_batch, 1, 1).to(device)
            out = model.predict(inp, dec_inp, src_att, trg_att)
            h=out[:,-1]
            dec_inp=torch.cat((dec_inp,torch.multinomial(h,1)),1)


        preds_tr_b=clusters[dec_inp[:,1:].cpu().numpy()].cumsum(1)+batch['src'][:,-1:,0:2].cpu().numpy()
        if sam == 0:
            multimodal_outputs[0] = [preds_tr_b[0], preds_tr_b[1:].transpose(1, 0, 2)]
        else:
            multimodal_outputs[sam] = [preds_tr_b[0], []]
    return multimodal_outputs


def load_predictor(args):
    mat = scipy.io.loadmat(os.path.join(args.dataset_folder, args.dataset_name, "clusters.mat"))
    clusters=mat['centroids']
    device = 'cuda:0'
    model=quantized_TF.QuantizedTF(clusters.shape[0], clusters.shape[0]+1, clusters.shape[0], N=args.layers,
                                   d_model=args.emb_size, d_ff=1024, h=args.heads).to(device)
    model.load_state_dict(torch.load(f'models/QuantizedTF/{args.name}/{args.epoch}.pth'))
    model.to(device)
    model.eval()
    return model, clusters


def get_predictions(args):
    """Get model predictions for each test scene and write the predictions in appropriate folders"""
    ## List of .json file inside the args.path (waiting to be predicted by the testing model)
    datasets = sorted([f.split('.')[-2] for f in os.listdir(args.path.replace('_pred', '')) if not f.startswith('.') and f.endswith('.ndjson')])

    ## Extract Model names from arguments and create its own folder in 'test_pred' for storing predictions
    ## WARNING: If Model predictions already exist from previous run, this process SKIPS WRITING
    for model in args.output:
        model_name = model.split('/')[-1].replace('.pkl', '')
        model_name = model_name + '_modes' + str(args.modes)

        ## Check if model predictions already exist
        if not os.path.exists(args.path):
            os.makedirs(args.path)
        if not os.path.exists(args.path + model_name):
            os.makedirs(args.path + model_name)
        else:
            print('Predictions corresponding to {} already exist.'.format(model_name))
            print('Loading the saved predictions')
            continue

        print("Model Name: ", model_name)
        model, clusters = load_predictor(args)
        goal_flag = False

        # Iterate over test datasets
        for dataset in datasets:
            # Load dataset
            dataset_name, scenes, scene_goals = load_test_datasets(dataset, goal_flag, args)

            # Get all predictions in parallel. Faster!
            scenes = tqdm(scenes)
            pred_list = Parallel(n_jobs=1)(delayed(predict_multimodal_scene)(model, clusters, paths, args)
                                           for (_, _, paths), scene_goal in zip(scenes, scene_goals))
            
            # Write all predictions
            write_predictions(pred_list, scenes, model_name, dataset_name, args)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--obs_length', default=9, type=int,
                        help='observation length')
    parser.add_argument('--pred_length', default=12, type=int,
                        help='prediction length')
    parser.add_argument('--write_only', action='store_true',
                        help='disable writing new files')
    parser.add_argument('--disable-collision', action='store_true',
                        help='disable collision metrics')
    parser.add_argument('--labels', required=False, nargs='+',
                        help='labels of models')
    parser.add_argument('--normalize_scene', action='store_true',
                        help='augment scenes')
    parser.add_argument('--modes', default=1, type=int,
                        help='number of modes to predict')

    # Transformer
    parser.add_argument('--dataset_folder',type=str,default='datasets/')
    parser.add_argument('--dataset_name',type=str,default='colfree_trajdata')
    parser.add_argument('--name', type=str, default="colfree_trajdata")
    parser.add_argument('--epoch',type=str,default="00095")
    parser.add_argument('--emb_size',type=int,default=512)
    parser.add_argument('--heads',type=int, default=8)
    parser.add_argument('--layers',type=int,default=6)
    args = parser.parse_args()

    scipy.seterr('ignore')

    args.output = ['traj_trans']
    args.path = args.dataset_folder + args.dataset_name + '/test_pred/'

    ## Writes to Test_pred
    ## Does NOT overwrite existing predictions if they already exist ###
    get_predictions(args)
    if args.write_only: # For submission to AICrowd.
        print("Predictions written in test_pred folder")
        exit()

    ## Evaluate using TrajNet++ evaluator
    trajnet_evaluate(args)


if __name__ == '__main__':
    main()
