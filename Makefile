@ kmeans:
	CUDA_VISIBLE_DEVICES=0 python kmeans_trajnetpp.py --dataset_name eth_data --sample 1.0
	CUDA_VISIBLE_DEVICES=0 python kmeans_trajnetpp.py --dataset_name hotel_data --sample 1.0
	CUDA_VISIBLE_DEVICES=0 python kmeans_trajnetpp.py --dataset_name zara1_data --sample 1.0
	CUDA_VISIBLE_DEVICES=0 python kmeans_trajnetpp.py --dataset_name zara2_data --sample 1.0
	CUDA_VISIBLE_DEVICES=0 python kmeans_trajnetpp.py --dataset_name univ_data --sample 1.0

	cp eth_data_1000_200000_scaleTrue_rotTrue/clusters.mat datasets/eth_data/
	cp hotel_data_1000_200000_scaleTrue_rotTrue/clusters.mat datasets/hotel_data/
	cp univ_data_1000_200000_scaleTrue_rotTrue/clusters.mat datasets/univ_data/
	cp zara1_data_1000_200000_scaleTrue_rotTrue/clusters.mat datasets/zara1_data/
	cp zara2_data_1000_200000_scaleTrue_rotTrue/clusters.mat datasets/zara2_data/

@ train_gpu0:
	CUDA_VISIBLE_DEVICES=0 python train_trajnetpp_quantizedTF.py --dataset_name eth_data --name eth_data --batch_size 1024 --max_epoch 50
	CUDA_VISIBLE_DEVICES=0 python train_trajnetpp_quantizedTF.py --dataset_name hotel_data --name hotel_data --batch_size 1024 --max_epoch 50
	CUDA_VISIBLE_DEVICES=0 python train_trajnetpp_quantizedTF.py --dataset_name univ_data --name univ_data --batch_size 1024 --max_epoch 50

@ train_gpu1:
	CUDA_VISIBLE_DEVICES=1 python train_trajnetpp_quantizedTF.py --dataset_name zara1_data --name zara1_data --batch_size 1024 --max_epoch 50
	CUDA_VISIBLE_DEVICES=1 python train_trajnetpp_quantizedTF.py --dataset_name zara2_data --name zara2_data --batch_size 1024 --max_epoch 50
