@ kmeans:
	CUDA_VISIBLE_DEVICES=0 python kmeans_trajnetpp.py --dataset_name eth_data
	cp eth_data_1000_200000_scaleTrue_rotTrue/clusters.mat datasets/eth_data/

	CUDA_VISIBLE_DEVICES=0 python kmeans_trajnetpp.py --dataset_name hotel_data
	cp hotel_data_1000_200000_scaleTrue_rotTrue/clusters.mat datasets/hotel_data/

	CUDA_VISIBLE_DEVICES=0 python kmeans_trajnetpp.py --dataset_name zara1_data
	cp zara1_data_1000_200000_scaleTrue_rotTrue/clusters.mat datasets/zara1_data/

	CUDA_VISIBLE_DEVICES=0 python kmeans_trajnetpp.py --dataset_name zara2_data
	cp zara2_data_1000_200000_scaleTrue_rotTrue/clusters.mat datasets/zara2_data/

	CUDA_VISIBLE_DEVICES=0 python kmeans_trajnetpp.py --dataset_name univ_data
	cp univ_data_1000_200000_scaleTrue_rotTrue/clusters.mat datasets/univ_data/

	# Create own validation set for TrajNet++
	# CUDA_VISIBLE_DEVICES=0 python kmeans_trajnetpp.py --dataset_name colfree_trajdata --obs 9
	# cp colfree_trajdata_1000_200000_scaleTrue_rotTrue/clusters.mat datasets/colfree_trajdata/


@ train_individual:
	CUDA_VISIBLE_DEVICES=1 python train_trajnetpp_individualTF.py --dataset_name eth_data --name eth_data --batch_size 1024
	CUDA_VISIBLE_DEVICES=1 python train_trajnetpp_individualTF.py --dataset_name univ_data --name univ_data --batch_size 1024
	CUDA_VISIBLE_DEVICES=1 python train_trajnetpp_individualTF.py --dataset_name zara2_data --name zara2_data --batch_size 1024
	
	# Create own validation set for TrajNet++
	# CUDA_VISIBLE_DEVICES=1 python train_trajnetpp_individualTF.py --dataset_name colfree_trajdata --name colfree_trajdata --batch_size 1024 --obs 9


@ train:
	CUDA_VISIBLE_DEVICES=1 python train_trajnetpp_quantizedTF.py --dataset_name eth_data --name eth_data --batch_size 1024
	CUDA_VISIBLE_DEVICES=1 python train_trajnetpp_quantizedTF.py --dataset_name univ_data --name univ_data --batch_size 1024
	CUDA_VISIBLE_DEVICES=1 python train_trajnetpp_quantizedTF.py --dataset_name zara2_data --name zara2_data --batch_size 1024
	
	# Create own validation set for TrajNet++
	# CUDA_VISIBLE_DEVICES=1 python train_trajnetpp_quantizedTF.py --dataset_name colfree_trajdata --name colfree_trajdata --batch_size 1024 --obs 9


@ test:
	CUDA_VISIBLE_DEVICES=1 python test_trajnetpp_quantizedTF.py --dataset_name eth_data --name eth_data --batch_size 1024 --epoch 00095
	CUDA_VISIBLE_DEVICES=1 python test_trajnetpp_quantizedTF.py --dataset_name univ_data --name univ_data --batch_size 1024 --epoch 00095
	CUDA_VISIBLE_DEVICES=1 python test_trajnetpp_quantizedTF.py --dataset_name zara2_data --name zara2_data --batch_size 1024 --epoch 00095

@ test_trajnet:  # For submission to AICrowd
	python -m trajnet_evaluator --write_only
