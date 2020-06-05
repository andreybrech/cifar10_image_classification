import datetime
import argparse
import os
from operator import attrgetter
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import model
from utils import evaluate, make_data, load_model_from_pretrained

def main(parser):
    args = parser.parse_args()
    if args.dataset_name != 'CIFAR10':
        raise NotImplemented
    print(f'args:{args}')
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.cuda else "cpu")

    # load dataset
    train_loader, test_loader = make_data(args, data_path='/data', num_workers=4)

    # configure network
    getter = attrgetter(args.network_type)
    nn_type = getter(model)
    net = nn_type(input_size=3*32*32, num_classes=10)
    net.to(device)
    print(net)
    # load from pretrained
    if args.from_pretrained:
        net = load_model_from_pretrained(args, net)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)

    # configure tensorboard writer
    name = '.'.join(str(datetime.datetime.now()).split(':'))
    writer = SummaryWriter(f'./logs/{name}')

    epochs = args.epochs
    running_loss = 0.0
    info_every = int(len(train_loader) / args.info_times_per_epoch)
    eval_every = int(len(train_loader) / args.eval_times_per_epoch)
    for epoch in range(epochs):
        net.train()
        for i, (data, labels) in enumerate(train_loader):
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            # print(data.shape,labels.shape)
            net_out = net(data)
            # print(net_out, labels)
            # print(net_out.shape,labels.shape)
            loss = criterion(net_out, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % info_every == info_every - 1:  # print every 400 mini-batches
                print('[%d, %5d / %5d] loss: %.3f' %
                      (epoch + 1, i + 1, len(train_loader), running_loss / info_every, ))
                running_loss = 0.0
            if i % eval_every == eval_every - 1:
                avg_loss, acc = evaluate(net, test_loader, criterion, device)
                print(f'avg_loss: {avg_loss}, acc: {acc}')
                writer.add_scalar('acc', acc, epoch * len(train_loader) + i)
                writer.add_scalar('avg_loss', avg_loss, epoch * len(train_loader) + i)
                writer.add_scalar('learning rate', scheduler.get_lr()[0], epoch * len(train_loader) + i)
                if args.save_model_on_eval:
                    if not 'models' in os.listdir('./'):
                        os.mkdir('models')
                    if not args.network_type in os.listdir('./models'):
                        os.mkdir(f'models/{args.network_type}')
                    PATH = f'./models/{args.network_type}/{args.dataset_name}_{args.network_type}_epoch_{epoch}_eval_{i // eval_every}.pth'
                    torch.save(net.state_dict(), PATH)

        if args.lr_sheduler:
            scheduler.step()
    avg_loss, acc = evaluate(net, test_loader, criterion, device)
    print(f'Finished Training. Avarage loss: {avg_loss}, accuracy: {acc}')
    # save model
    if args.save_model:
        PATH = f'./models/{args.network_type}/{args.dataset_name}_{args.network_type}_final.pth'
        torch.save(net.state_dict(), PATH)


if __name__ == '__main__':
    # training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', choices=['CIFAR10'], default='CIFAR10',
                        help='choose dataset name. Available:[CIFAR10]')
    parser.add_argument('--batch_size_train', type=int, default=16,
                        help="batch size for training (default:16)")
    parser.add_argument('--batch_size_test', type=int, default=16,
                        help="batch size for testing (default:16)")
    parser.add_argument('--epochs', type=int, default=5,
                        help="epochs number for training (default:5)")
    parser.add_argument('--lr', type=float, default=0.01,
                        help="learning rate (default:0.01)")
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='enables CUDA training')
    parser.add_argument('--lr_sheduler', action='store_true', default=False,
                        help='enables lr_sheduler training')
    parser.add_argument('--network_type', choices=['CNN', 'MLP'], default='CNN',
                        help='choose model type name. Available: [CNN, MLP]')
    parser.add_argument('--save_model', action='store_true', default=False,
                        help='enables saving model')
    parser.add_argument('--save_model_on_eval', action='store_true', default=True,
                        help='enables saving model on every evaluation')
    parser.add_argument('--info_times_per_epoch',  type=int, default=4,
                        help='chose frequency of info per epoch ')
    parser.add_argument('--eval_times_per_epoch', type=int, default=4,
                        help='chose frequency of evaluation per epoch ')
    parser.add_argument('--from_pretrained', action='store_true', default=True,
                        help='train model from scratch (choose True) of load pretrained weigts(choose False)(default:False)')
    parser.add_argument('--from_pretrained_epoch', type=int, default=-1,
                        help='chose epoch of pretrained weights')
    parser.add_argument('--from_pretrained_eval', type=int, default=1,
                        help='chose eval number of pretrained weights. Available if used --from_pretrained_epoch')
    main(parser)