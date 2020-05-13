from __future__ import print_function, division

import sys
import time
import torch

from .util import AverageMeter, accuracy
from .ntk_util import generate_partial_ntk


def train_vanilla(epoch, train_loader, model, criterion, optimizer, opt):
    """vanilla training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    for idx, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        input = input.float()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()

        # ===================forward=====================
        output, _ = model(input) # free to ignore jvp here
        loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # tensorboard logger
        pass

        # print info
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]	'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})	'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})	'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})	'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})	'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, idx, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
            sys.stdout.flush()

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg, losses.avg


def train_distill(epoch, train_loader, module_list, criterion_list, optimizer, opt):
    """One epoch distillation"""
    # set modules as train()
    for module in module_list:
        module.train()
    # set teacher as eval()
    module_list[-1].eval()

    if opt.distill == 'abound':
        module_list[1].eval()
    elif opt.distill == 'factor':
        module_list[2].eval()

    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]
    criterion_kd = criterion_list[2] # This has to be Frobenius Norm

    model_s = module_list[0]
    model_t = module_list[-1]

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()

    # hard-coded
    batch_size = opt.batch_size
    n_cls = 100
    ntk_s = torch.zeros(batch_size, batch_size, n_cls, n_cls, requires_grad=True).cuda()
    ntk_t = torch.zeros(batch_size, batch_size, n_cls, n_cls, requires_grad=True).cuda()
    # ntk_loss = torch.zeros(1, requires_grad=True).cuda()
    # torch.autograd.set_detect_anomaly(True)

    for idx, data in enumerate(train_loader):
        if opt.distill in ['crd']:
            input, target, index, contrast_idx = data
        else:
            input, target, index = data
        data_time.update(time.time() - end)

        input = input.float()

        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()
            index = index.cuda()
            if opt.distill in ['crd']:
                contrast_idx = contrast_idx.cuda()

        # ===================forward=====================
        preact = False # Since support for 'abound' method has not been added yet

        # Start with zero grad
        optimizer.zero_grad()

        # NOTE: Support only for kd, crd and ntk
        if opt.distill in ['crd']:
            feat_s, logit_s, _ = model_s(input, is_feat=True, preact=False)
            with torch.no_grad():
                feat_t, logit_t, _ = model_t(input, is_feat=True, preact=False)
                feat_t = [f.detach() for f in feat_t]

            # cls + kl div
            loss_cls = criterion_cls(logit_s, target)
            loss_div = criterion_div(logit_s, logit_t)

            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd = criterion_kd(f_s, f_t, index, contrast_idx)
            loss = opt.gamma * loss_cls + opt.alpha * loss_div + opt.beta * loss_kd

        elif opt.distill in ['ntk']:
            num_proj = 100 # Set number of projections
            ntk_diff = 0
            ntk_loss = torch.zeros(1, requires_grad=True).cuda()

            for p in range(num_proj):
                logit_s, jvp_s = model_s(input) #is_feat=False, preact=False
                with torch.no_grad():
                    logit_t, jvp_t = model_t(input)

                # jvp is of shape batch_size x n_classes
                ntk_s = generate_partial_ntk(jvp_s, ntk_s)
                ntk_t = generate_partial_ntk(jvp_t, ntk_t)
                ntk_loss = (ntk_s - ntk_t).sum() # NOTE: What exactly should I square here?

                # save elementwise difference
                # ntk_diff += (ntk_s - ntk_t).sum()
                # ntk_diff.detach()

                # Are the following retain grads required?
                ntk_loss.retain_grad()
                jvp_s.retain_grad()
                ntk_s.retain_grad()
                ntk_loss.backward(retain_graph=True)

                # WARNING: Need to detach all random variables..clear the graph here

            # NOTE: Multiply sum of element wise difference to the grads.

            # cls + kl div
            loss_cls = criterion_cls(logit_s, target)
            loss_div = criterion_div(logit_s, logit_t)
            loss = (opt.gamma/opt.beta) * loss_cls + (opt.alpha/opt.beta) * loss_div

        acc1, acc5 = accuracy(logit_s, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # ===================loss-backward=====================

        loss.backward()
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]	'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})	'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})	'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})	'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})	'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, idx, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))
            sys.stdout.flush()

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg, losses.avg





def validate(val_loader, model, criterion, opt):
    """validation"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for idx, (input, target) in enumerate(val_loader):

            input = input.float()
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()

            # compute output
            output, _ = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]	'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})	'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})	'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})	'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       idx, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg, top5.avg, losses.avg