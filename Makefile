@ kmeans:
	CUDA_VISIBLE_DEVICES=0 python kmeans_trajnetpp.py --dataset_name eth_data
	CUDA_VISIBLE_DEVICES=0 python kmeans_trajnetpp.py --dataset_name hotel_data
	CUDA_VISIBLE_DEVICES=0 python kmeans_trajnetpp.py --dataset_name zara1_data
	CUDA_VISIBLE_DEVICES=0 python kmeans_trajnetpp.py --dataset_name zara2_data
	CUDA_VISIBLE_DEVICES=0 python kmeans_trajnetpp.py --dataset_name univ_data
	CUDA_VISIBLE_DEVICES=0 python kmeans_trajnetpp.py --dataset_name colfree_trajdata --obs 9

	###### Copy clusters.mat into respective folders
	cp eth_data_1000_200000_scaleTrue_rotTrue/clusters.mat datasets/eth_data/
	cp hotel_data_1000_200000_scaleTrue_rotTrue/clusters.mat datasets/hotel_data/
	cp univ_data_1000_200000_scaleTrue_rotTrue/clusters.mat datasets/univ_data/
	cp zara1_data_1000_200000_scaleTrue_rotTrue/clusters.mat datasets/zara1_data/
	cp zara2_data_1000_200000_scaleTrue_rotTrue/clusters.mat datasets/zara2_data/
	cp colfree_trajdata_1000_200000_scaleTrue_rotTrue/clusters.mat datasets/colfree_trajdata/


@ train:
	CUDA_VISIBLE_DEVICES=1 python train_trajnetpp_quantizedTF.py --dataset_name eth_data --name eth_data --batch_size 1024
	CUDA_VISIBLE_DEVICES=1 python train_trajnetpp_quantizedTF.py --dataset_name univ_data --name univ_data --batch_size 1024
	CUDA_VISIBLE_DEVICES=1 python train_trajnetpp_quantizedTF.py --dataset_name zara2_data --name zara2_data --batch_size 1024
	CUDA_VISIBLE_DEVICES=1 python train_trajnetpp_quantizedTF.py --dataset_name colfree_trajdata --name colfree_trajdata --batch_size 1024 --obs 9

@ test:
	CUDA_VISIBLE_DEVICES=1 python test_trajnetpp_quantizedTF.py --dataset_name eth_data --name eth_data --batch_size 1024 --epoch 00095
	CUDA_VISIBLE_DEVICES=1 python test_trajnetpp_quantizedTF.py --dataset_name univ_data --name univ_data --batch_size 1024 --epoch 00095
	CUDA_VISIBLE_DEVICES=1 python test_trajnetpp_quantizedTF.py --dataset_name zara2_data --name zara2_data --batch_size 1024 --epoch 00095
