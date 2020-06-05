# cifar10_cnn_and_mlp

cifar-10 image classification using pytorch

in project used two types of network:
 - fully connected
 - convolutional
 
 
# Dependecies
torch==1.4.0
torchvision==0.5.0


# How to install
$ cd myproject  
$ git init  
$ git add .  
$ git commit  
$ git clone --bare myproject  git@github.com:andreybrech/neural_network_project.git
$ pip install -r requirements.txt


# Models
1. 3 layer MLP
2. CNN with 2 convolution layers and with 2 dense layers


# Train
usage: train.py [-h] [--dataset_name {CIFAR10}]
                [--batch_size_train BATCH_SIZE_TRAIN]
                [--batch_size_test BATCH_SIZE_TEST] [--epochs EPOCHS]
                [--lr LR] [--cuda] [--lr_sheduler] [--network_type {CNN,MLP}]
                [--save_model] [--save_model_on_eval]
                [--info_times_per_epoch INFO_TIMES_PER_EPOCH]
                [--eval_times_per_epoch EVAL_TIMES_PER_EPOCH]
                [--from_pretrained]
                [--from_pretrained_epoch FROM_PRETRAINED_EPOCH]
                [--from_pretrained_eval FROM_PRETRAINED_EVAL]

optional arguments:
  -h, --help            show this help message and exit
  --dataset_name {CIFAR10}
                        choose dataset name. Available:[CIFAR10]
  --batch_size_train BATCH_SIZE_TRAIN
                        batch size for training (default:16)
  --batch_size_test BATCH_SIZE_TEST
                        batch size for testing (default:16)
  --epochs EPOCHS       epochs number for training (default:5)
  --lr LR               learning rate (default:0.01)
  --cuda                enables CUDA training
  --lr_sheduler         enables lr_sheduler training
  --network_type {CNN,MLP}
                        choose model type name. Available: [CNN, MLP]
  --save_model          enables saving model
  --save_model_on_eval  enables saving model on every evaluation
  --info_times_per_epoch INFO_TIMES_PER_EPOCH
                        chose frequency of info per epoch
  --eval_times_per_epoch EVAL_TIMES_PER_EPOCH
                        chose frequency of evaluation per epoch
  --from_pretrained     train model from scratch (choose True) of load
                        pretrained weigts(choose False)(default:False)
  --from_pretrained_epoch FROM_PRETRAINED_EPOCH
                        chose epoch of pretrained weights
  --from_pretrained_eval FROM_PRETRAINED_EVAL
                        chose eval number of pretrained weights. Available if
                        used --from_pretrained_epoch
# How to train  from scratch:
$python train.py --network_type "CNN"

# How to train  from pretrained
$python train.py --network_type "CNN" --from_pretrained

# How to train  from pretrained (specify epoch and evaluation number)
$python train.py --network_type "CNN" --from_pretrained --from_pretrained_epoch 1 --from_pretrained_eval 1


# Test
usage: test.py [-h] [--dataset_name {CIFAR10}]
               [--batch_size_train BATCH_SIZE_TRAIN]
               [--batch_size_test BATCH_SIZE_TEST] [--cuda]
               [--network_type {CNN,MLP}] [--path_to_model PATH_TO_MODEL]
               [--from_pretrained_epoch FROM_PRETRAINED_EPOCH]
               [--from_pretrained_eval FROM_PRETRAINED_EVAL]

optional arguments:
  -h, --help            show this help message and exit
  --dataset_name {CIFAR10}
                        choose dataset name. Available:[CIFAR10]
  --batch_size_train BATCH_SIZE_TRAIN
                        batch size for training (default:16)
  --batch_size_test BATCH_SIZE_TEST
                        batch size for testing (default:16)
  --cuda                enables CUDA training
  --network_type {CNN,MLP}
                        choose model type name. Available: [CNN, MLP]
  --path_to_model PATH_TO_MODEL
                        chose frequency of info per epoch
  --from_pretrained_epoch FROM_PRETRAINED_EPOCH
                        chose epoch of pretrained weights
  --from_pretrained_eval FROM_PRETRAINED_EVAL
                        chose eval number of pretrained weights. Available if
                        used --from_pretrained_epoch               
   
# How to test (model from last epoch of final model)
$python test.py --network_type "CNN" 

# How to test  (specify epoch and evaluation number)
$python test.py --network_type "CNN" --from_pretrained_epoch 1 --from_pretrained_eval 1

# results
| CNN accuracy| MLP accuracy|
| ------------- | ------------- |
| 0.6736 | 0.532  |
