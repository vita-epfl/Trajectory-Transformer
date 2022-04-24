MODES=3

# Run K-means clustering
CUDA_VISIBLE_DEVICES=0 python kmeans_trajnetpp.py \
    --dataset_name colfree_trajdata --obs 9 

cp colfree_trajdata_1000_200000_scaleTrue_rotTrue/clusters.mat datasets/colfree_trajdata/

# Train the model 
CUDA_VISIBLE_DEVICES=1 python train_trajnetpp_quantizedTF.py \
    --dataset_name colfree_trajdata --name colfree_trajdata \
    --obs 9 --batch_size 128 --max_epoch 100

# Evaluate on Trajnet++ 
python -m trajnet_evaluator \
    --dataset_name colfree_trajdata --write_only --modes ${MODES} 
