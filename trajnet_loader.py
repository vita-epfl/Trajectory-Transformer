import numpy as np
import torch

import trajnetplusplustools


def pre_process_test(sc_, obs_len=8):
    obs_frames = [primary_row.frame for primary_row in sc_[0]][:obs_len]
    last_frame = obs_frames[-1]
    sc_ = [[row for row in ped] for ped in sc_ if ped[0].frame <= last_frame]
    return sc_


def trajnet_loader(data_loader, args):
    batch = {'src': [], 'trg': []}
    num_batches = 0
    for batch_idx, (filename, scene_id, paths) in enumerate(data_loader):
        ## make new scene
        pos_scene = trajnetplusplustools.Reader.paths_to_xy(paths)[:, 0]  # primary ped
        vel_scene = np.zeros_like(pos_scene)
        vel_scene[1:] = pos_scene[1:] - pos_scene[:-1]
        attr_scene = np.concatenate((pos_scene, vel_scene), axis=1)
        batch['src'].append(attr_scene[:args.obs])
        batch['trg'].append(attr_scene[-args.preds:])
        num_batches += 1

        if (num_batches % args.batch_size != 0) and (batch_idx + 1 != len(data_loader)):
            continue
        
        batch['src'] = torch.Tensor(np.stack(batch['src']))
        batch['trg'] = torch.Tensor(np.stack(batch['trg']))

        yield batch
        batch = {'src': [], 'trg': []}


def trajnet_test_loader(data_loader, args):
    batch = {'src': [], 'trg': []}
    seq_start_end = []
    num_batches = 0
    for batch_idx, (filename, scene_id, paths) in enumerate(data_loader):
        ## make new scene
        paths = pre_process_test(paths, args.obs)
        pos_scene = trajnetplusplustools.Reader.paths_to_xy(paths)
        vel_scene = np.zeros_like(pos_scene)
        vel_scene[1:] = pos_scene[1:] - pos_scene[:-1]
        attr_scene = np.concatenate((pos_scene, vel_scene), axis=2)
        seq_start_end.append(pos_scene.shape[1])
        batch['src'].append(attr_scene[:args.obs])
        batch['trg'].append(attr_scene[-args.preds:])
        num_batches += 1

        if (num_batches % args.batch_size != 0) and (batch_idx + 1 != len(data_loader)):
            continue
        
        batch['src'] = torch.Tensor(np.concatenate(batch['src'], axis=1)).permute(1, 0, 2)
        batch['trg'] = torch.Tensor(np.concatenate(batch['trg'], axis=1)).permute(1, 0, 2)
        seq_start_end = [0] + seq_start_end
        seq_start_end = torch.LongTensor(np.array(seq_start_end).cumsum())
        seq_start_end = torch.stack((seq_start_end[:-1], seq_start_end[1:]), dim=1)

        yield batch, seq_start_end
        batch = {'src': [], 'trg': []}
        seq_start_end = []
