# neural_network_project
simple project of cifar-10 image classification using pytorch

in project used two types of network:
 - fully connected
 - convolutional
 

Model using argparse to configure parameters from console.
Argparse parameters:
  --batch_size_train - batch size for training (default:16)
  --batch_size_test - batch size for testing (default:16)
  --epochs - epochs number for training (default:5)
  --lr - learning rate (default:0.01)
  --cuda -enables CUDA training
  --lr_sheduler - enables lr_sheduler training
  --network_type - 1: CNN; 2:MLP
  --save_model - enables saving model
  --detailed_statistics -  enables detailed statistics: accuracy, avg_loss 4 time per epoch and write it to tensorboard
