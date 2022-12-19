import logging
import time
from collections import OrderedDict

from Evaluate import *
import torch, gc


def train(net, train_dataloader, test_dataloader, optimizer, loss_function, total_epoch, lr, args,
          train_sampler, num_classes, use_aux, size=(224, 224), pretrained=None, starting_epoch=0):
    logging.basicConfig(filename="logs/VOC_0.log",
                        filemode='w',
                        level=logging.INFO,
                        format="%(asctime)s %(levelname)s: %(msg)s",
                        datefmt="%Y-%m-%d %H:%M:%S")

    gc.collect()
    torch.cuda.empty_cache()
    logging.info('Cityscapes dataset using net : FCN')
    logging.info('using batch_size: {} use_aux:{}'.format(train_dataloader.batch_size, use_aux))
    logging.info('total_epoch:{} lr:{} optimizer:{}'.format(total_epoch, lr, optimizer))
    loss_function = loss_function.cuda()
    t1 = time.time()
    if pretrained:
        x = torch.load(pretrained, map_location=torch.device(args.local_rank))
        y = OrderedDict()
        for k, v in x.items():
            t = 'module.' + k
            y[t] = v
        net.load_state_dict(y)
    best = 0
    for i in range(starting_epoch + 1, total_epoch + 1, 1):
        net.train()
        train_sampler.set_epoch(i)
        total_loss = 0
        # hist = np.zeros((num_classes, num_classes))
        for imgs, labels in train_dataloader:
            imgs = imgs.cuda(args.local_rank, non_blocking=True)
            labels = labels.clone().detach().cuda(args.local_rank, non_blocking=True)
            # if use_aux:
            #     aux_out, out = net(imgs)
            #     aux_loss = loss_function(aux_out, labels)
            #     master_loss = loss_function(out, labels)
            #     loss = aux_loss * 0.4 + master_loss
            # else:
            out = net(imgs)
            loss = loss_function(out, labels)
            # preds = torch.argmax(out, dim=1).squeeze().cpu()
            # labels = labels.cpu()
            # hist += calc_hist(labels, preds, num_classes)
            total_loss += loss.cpu().item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # iou = np.diag(hist) / (np.sum(hist, axis=0) + np.sum(hist, axis=1) - np.diag(hist))
        # miou = np.nanmean(iou)
        mean_loss = total_loss / len(train_dataloader)

        if args.local_rank == 0:
            logging.info("epoch: %d training loss: %f" % (i, mean_loss))
            print("epoch: %d  training loss: %f" % (i, mean_loss))
            # logging.info("epoch: %d training loss: %f miou %f" % (i, mean_loss, miou))
            # print("epoch: %d  training loss: %f miou %f" % (i, mean_loss, miou))
            net.eval()
            with torch.no_grad():
                hist = np.zeros((num_classes, num_classes))
                true_pixel = 0
                for imgs, labels in test_dataloader:
                    imgs = imgs.cuda(non_blocking=True)
                    labels = labels.cuda(non_blocking=True)
                    k = imgs.shape[0]
                    # if use_aux:
                    #     _, out = net(imgs)
                    # else:
                    out = net(imgs)
                    preds = torch.argmax(out, dim=1).squeeze().cpu()
                    labels = labels.cpu()
                    hist += calc_hist(labels, preds, num_classes)
                    for p, l in zip(preds, labels):
                        true_pixel += np.sum(p.numpy() == l.numpy())
                iou = np.diag(hist) / (np.sum(hist, axis=0) + np.sum(hist, axis=1) - np.diag(hist))
                miou = np.nanmean(iou)
                if miou > best:
                    best = miou
                if i % 10 == 0:
                    print(iou)

                acc = true_pixel / (k * len(test_dataloader) * size[0] * size[1])
                logging.info("epoch: %d miou: %f acc: %f" % (i, miou, acc))

        if i % 40 == 0 and args.local_rank == 0:
            torch.save(net.module.state_dict(), "./model/MyFCN{}.pth".format(i),
                       _use_new_zipfile_serialization=False)

    if args.local_rank == 0:
        t2 = time.time()
        print("total time: %f" % (t2 - t1))
        print('mean epoch time: %f' % ((t2 - t1) / (total_epoch - starting_epoch)))
        logging.info(iou)
        logging.info("total time: %f" % (t2 - t1))
        logging.info('mean epoch time: %f' % ((t2 - t1) / (total_epoch - starting_epoch)))
        logging.info('best miou : {}'.format(best))


if __name__ == '__main__':
    pass
