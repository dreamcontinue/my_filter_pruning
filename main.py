import os, sys, time, random,torch
import models
import torch.backends.cudnn as cudnn
import argparse
import dataset
import compute_flops as flops
# import mask,mask_modify
import mask_modify as mask
from utils import AverageMeter, \
    RecorderMeter, time_string, \
    convert_secs2time,print_log,\
    accuracy,adjust_learning_rate,\
    save_checkpoint

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))
print(model_names)
parser=argparse.ArgumentParser(description='filter pruning',
                               formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', type=str, default='cifar10',
                    choices=['cifar10', 'cifar100', 'imagenet'],
                    help='Choose between Cifar10/100 and ImageNet.')
parser.add_argument('--arch', metavar='ARCH', default='vgg16_cifar', choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: resnext29_8_64)')
# Optimization options
parser.add_argument('--epochs', type=int, default=300, help='Nu mber of epochs to train.')
parser.add_argument('--batch_size', type=int, default=256, help='Batch size.')
parser.add_argument('--learning_rate', type=float, default=0.1, help='The Learning Rate.')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', type=float, default=0.0005, help='Weight decay (L2 penalty).')
parser.add_argument('--schedule', type=int, nargs='+', default=[120, 225, 275],
                    help='Decrease learning rate at these epochs.')
parser.add_argument('--gammas', type=float, nargs='+', default=[0.1, 0.1, 0.5],
                    help='LR is multiplied by gamma on schedule, number of gammas should be equal to schedule')
# Checkpoints
parser.add_argument('--save_path', type=str, default='./', help='Folder to save checkpoints and log.')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
# Acceleration
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--workers', type=int, default=2, help='number of data loading workers (default: 2)')
# random seed
parser.add_argument('--manualSeed', type=int, default=10086, help='manual seed')
# compress rate
parser.add_argument('--rate', type=float, default=0.8, help='compress rate of model')
parser.add_argument('--epoch_prune', type=int, default=1, help='compress layer of model')
parser.add_argument('--use_state_dict', dest='use_state_dict', action='store_true', help='use state dcit or not')
parser.add_argument('--description',type=str,default='')
parser.add_argument('--last_index',type=int,default=0)

args=parser.parse_args()
args.use_cuda = args.ngpu > 0 and torch.cuda.is_available()

if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if args.use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)
cudnn.benchmark = True
# custom args
# args.learning_rate=0.001 # for vgg
args.arch='resnet32'
args.dataset='cifar100'
args.way='my'
args.rate=0.7
args.description='{}_{}_{}_{}'.format(args.arch,args.dataset,args.way,args.rate)
args.save_path='./{}/'.format(args.description)
args.resume=args.save_path+'ckpt/'
args.model_save=args.save_path+'model.pth'
args.last_index=90
args.original_train=False
args.original_model_save=args.save_path+'original_model.pth'
#
def main():
    # Init logger
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    if not os.path.isdir(args.resume):
        os.makedirs(args.resume)
    log = open(os.path.join(args.save_path, '{}.txt'.format(args.description)), 'w')
    print_log('save path : {}'.format(args.save_path), log)
    state = {k: v for k, v in args._get_kwargs()}
    print_log(state, log)
    print_log("Random Seed: {}".format(args.manualSeed), log)
    print_log("use cuda: {}".format(args.use_cuda), log)
    print_log("python version : {}".format(sys.version.replace('\n', ' ')), log)
    print_log("torch  version : {}".format(torch.__version__), log)
    print_log("cudnn  version : {}".format(torch.backends.cudnn.version()), log)
    print_log("Compress Rate: {}".format(args.rate), log)
    print_log("Epoch prune: {}".format(args.epoch_prune), log)
    print_log("description: {}".format(args.description), log)

    # Init data loader
    if args.dataset=='cifar10':
        train_loader=dataset.cifar10DataLoader(True,args.batch_size,True,args.workers)
        test_loader=dataset.cifar10DataLoader(False,args.batch_size,False,args.workers)
        num_classes=10
    elif args.dataset=='cifar100':
        train_loader=dataset.cifar100DataLoader(True,args.batch_size,True,args.workers)
        test_loader=dataset.cifar100DataLoader(False,args.batch_size,False,args.workers)
        num_classes=100
    elif args.dataset=='imagenet':
        assert False,'Do not finish imagenet code'
    else:
        assert False,'Do not support dataset : {}'.format(args.dataset)

    # Init model
    if args.arch=='cifarvgg16':
        net=models.vgg16_cifar(True,num_classes)
    elif args.arch=='resnet32':
        net=models.resnet32(num_classes)
    elif args.arch=='resnet110':
        net=models.resnet110(num_classes)
    else:
        assert False,'Not finished'


    print_log("=> network:\n {}".format(net),log)
    net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

    # define loss function (criterion) and optimizer
    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(net.parameters(), state['learning_rate'], momentum=state['momentum'],
                                weight_decay=state['decay'], nesterov=True)

    if args.use_cuda:
        net.cuda()
        criterion.cuda()

    recorder = RecorderMeter(args.epochs)
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print_log("=> loading checkpoint '{}'".format(args.resume), log)
            checkpoint = torch.load(args.resume)
            recorder = checkpoint['recorder']
            args.start_epoch = checkpoint['epoch']
            if args.use_state_dict:
                net.load_state_dict(checkpoint['state_dict'])
            else:
                net = checkpoint['state_dict']
            optimizer.load_state_dict(checkpoint['optimizer'])
            print_log("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']), log)

            if args.evaluate:
                time1=time.time()
                validate(test_loader,net,criterion,args.use_cuda,log)
                time2=time.time()
                print('validate function took %0.3f ms' % ((time2 - time1) * 1000.0))
                return
        else:
            print_log("=> no checkpoint found at '{}'".format(args.resume), log)
    else:
        print_log("=> not use any checkpoint for {} model".format(args.description), log)

    if args.original_train:
        start_time=time.time()
        epoch_time=AverageMeter()
        for epoch in range(args.start_epoch,args.epochs):
            current_learning_rate=adjust_learning_rate(args.learning_rate,optimizer,epoch,args.gammas,args.schedule)
            need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * (args.epochs - epoch))
            need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs)
            print_log(
                '\n==>>{:s} [Epoch={:03d}/{:03d}] {:s} [learning_rate={:6.4f}]'.format(time_string(), epoch, args.epochs,
                                                                                       need_time, current_learning_rate) \
                + ' [Best : Accuracy={:.2f}]'.format(recorder.max_accuracy(False)), log)
            train_acc, train_los = train(train_loader, net, criterion, optimizer, epoch, args.use_cuda, log)
            val_acc_2, val_los_2 = validate(test_loader, net, criterion, args.use_cuda, log)
            recorder.update(epoch, train_los, train_acc, val_los_2, val_acc_2)

            epoch_time.update(time.time() - start_time)
            start_time = time.time()
        torch.save(net, args.original_model_save)
        return


    comp_rate=args.rate
    m=mask.Mask(net,comp_rate,args.last_index)
    print("-" * 10 + "one epoch begin" + "-" * 10)
    print("the compression rate now is %f" % comp_rate)

    val_acc_1, val_los_1 = validate(test_loader, net, criterion, args.use_cuda,log)
    print(" accu before is: %.3f %%" % val_acc_1)

    m.model=net
    print('before pruning')
    m.print_weights_zero()
    m.init_mask_row(args.use_cuda)
    m.do_mask()
    print('after pruning')
    m.print_weights_zero()
    net=m.model#update net

    if args.use_cuda:
        net=net.cuda()
    val_acc_2, val_los_2 = validate(test_loader, net, criterion, args.use_cuda,log)
    print(" accu after is: %.3f %%" % val_acc_2)
    #

    start_time=time.time()
    epoch_time=AverageMeter()
    for epoch in range(args.start_epoch,args.epochs):
        current_learning_rate=adjust_learning_rate(args.learning_rate,optimizer,epoch,args.gammas,args.schedule)
        need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * (args.epochs - epoch))
        need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs)
        print_log(
            '\n==>>{:s} [Epoch={:03d}/{:03d}] {:s} [learning_rate={:6.4f}]'.format(time_string(), epoch, args.epochs,
                                                                                   need_time, current_learning_rate) \
            + ' [Best : Accuracy={:.2f}]'.format(recorder.max_accuracy(False)), log)
        train_acc,train_los=train(train_loader,net,criterion,optimizer,epoch,args.use_cuda,log)
        if (epoch % args.epoch_prune == 0 or epoch == args.epochs - 1):
            m.model=net
            print('before pruning')
            m.print_weights_zero()
            m.init_mask(args.use_cuda)
            m.do_mask()
            print('after pruning')
            m.print_weights_zero()
            net=m.model
            if args.use_cuda:
                net=net.cuda()

        val_acc_2, val_los_2 = validate(test_loader, net, criterion,args.use_cuda,log)

        is_best = recorder.update(epoch, train_los, train_acc, val_los_2, val_acc_2)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': net,
            'recorder': recorder,
            'optimizer': optimizer.state_dict(),
        }, is_best, args.resume, 'checkpoint.pth.tar')
        print('save ckpt done')

        epoch_time.update(time.time()-start_time)
        start_time=time.time()
    torch.save(net,args.model_save)
    # torch.save(net,args.save_path)
    flops.print_model_param_nums(net)
    flops.count_model_param_flops(net,32,False)
    log.close()

def train(train_loader,model,criterion,optimizer,epoch,use_cuda=False,log=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.train()

    end_time=time.time()
    for i, (input, label) in enumerate(train_loader):
        data_time.update(time.time() - end_time)
        if use_cuda:
            label = label.cuda()
            input = input.cuda()
        with torch.no_grad():
            input_var=torch.autograd.Variable(input)
            label_var=torch.autograd.Variable(label)
        output=model(input_var)
        loss=criterion(output,label_var)
        prec1, prec5 = accuracy(output.data, label, topk=(1, 5))
        # torch>=0.5
        losses.update(loss.data, input.size(0))
        top1.update(prec1, input.size(0))
        top5.update(prec5, input.size(0))
        # torch<0.5
        # losses.update(loss.data[0], input.size(0))
        # ...
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time()-end_time)
        end_time=time.time()

    print_log('  **Train** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(top1=top1, top5=top5),log)
    return top1.avg, losses.avg

def validate(val_loader,model,criterion,use_cuda=False,log=None):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.eval()
    for i,(input,label) in enumerate(val_loader):
        if use_cuda:
            label=label.cuda()
            input=input.cuda()
        with torch.no_grad():
            input_var=torch.autograd.Variable(input)
            label_var=torch.autograd.Variable(label)
        output=model(input_var)
        loss=criterion(output,label_var)
        prec1, prec5 = accuracy(output.data, label, topk=(1, 5))
        # torch>=0.5
        losses.update(loss.data, input.size(0))
        top1.update(prec1, input.size(0))
        top5.update(prec5, input.size(0))
        # torch<0.5
        # losses.update(loss.data[0], input.size(0))
        # ...
    print_log('  **Test** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(top1=top1, top5=top5),log)
    return top1.avg, losses.avg

if __name__ == '__main__':
    main()
