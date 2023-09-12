import os
import random
import datetime
import dateutil.tz
import argparse
import numpy as np
from bdpy.util import makedir_ifnot
import torch
from torch import optim
from torch.backends import cudnn
from torch.nn import CrossEntropyLoss
from scipy.stats import pearsonr

from dataset.datasets import BrainDataset
from model.models import BrainDecoder


def parse_args():
    parser = argparse.ArgumentParser(description='Train Brain Decoder')
    parser.add_argument('--fmri-train-file', dest='fmri_train_path', type=str)
    parser.add_argument('--img-fea-train-file', dest='img_fea_train_path', type=str)
    parser.add_argument('--fmri-test-file', dest='fmri_test_path', type=str)
    parser.add_argument('--img-fea-test-file', dest='img_fea_test_path', type=str)
    parser.add_argument('--model-dir', type=str)

    parser.add_argument('--CUDA', action='store_true', default=False)
    parser.add_argument('--gpu-id', type=int, default=-1)
    parser.add_argument('--manualSeed', type=int, default=100, help='manual seed')
    parser.add_argument('--batch-size', type=int, default=10)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--learning-rate', type=float, default=2e-4)
    parser.add_argument('--weight-decay', type=float, default=0)
    parser.add_argument('--max-epoch', type=int, default=20)
    parser.add_argument('--snapshot-interval', type=int, default=1)

    parser.add_argument('--average', action='store_true', default=False,
                        help='taking the average across all training trials under the same stimulus')
    parser.add_argument('--temp1', type=float, default=1.0)
    parser.add_argument('--temp2', type=float, default=1.0)
    parser.add_argument('--lambda1', type=float, default=0.05)
    parser.add_argument('--lambda2', type=float, default=0.05)
    args = parser.parse_args()
    return args


def train(dataloader, brain_decoder, optimizer, epoch, args):
    brain_decoder.train()

    for step, data in enumerate(dataloader):
        optimizer.zero_grad()
        fmri_data, img_fea = data
        # fmri_data:  bs * trial_num * dim_fmri
        # img_fea:    bs * dim_fea

        if args.CUDA:
            fmri_data = fmri_data.to(torch.float32).cuda()
            img_fea = img_fea.to(torch.float32).cuda()
        else:
            fmri_data = fmri_data.to(torch.float32)
            img_fea = img_fea.to(torch.float32)

        if args.average:
            fmri_data = torch.mean(fmri_data, dim=1)
            fmri_fea = brain_decoder(fmri_data)
            batch_size = fmri_fea.shape[0]
            total_loss = torch.nn.MSELoss()(fmri_fea, img_fea).log()
            if (step + 1) % (len(dataloader) // 3) == 0:
                cur_time = datetime.datetime.now(dateutil.tz.tzlocal()).strftime('%Y_%m_%d %H:%M:%S')
                print('{} epoch {} | {}/{} step | mse {:.4f}'.format(cur_time, epoch + 1, (step + 1) * batch_size,
                                                                     len(dataloader) * batch_size, total_loss))
        else:
            fmri_fea = brain_decoder(fmri_data)
            batch_size, trial_num, dim_fea = fmri_fea.shape
            device = fmri_fea.device

            fmri2img_similarity_matrix = torch.cosine_similarity(
                fmri_fea.view(batch_size * trial_num, -1).unsqueeze(1),
                img_fea.unsqueeze(0),
                dim=-1
            ) / args.temp1
            fmri2img_labels = torch.arange(batch_size, device=device).unsqueeze(1).expand(batch_size,
                                                                                          trial_num).contiguous().view(-1)
            fmri2img_cont_loss = CrossEntropyLoss()(fmri2img_similarity_matrix, fmri2img_labels)

            fmri2fmri_similarity_matrix = torch.cosine_similarity(
                fmri_fea.view(batch_size * trial_num, -1).unsqueeze(1),
                fmri_fea.view(batch_size * trial_num, -1).unsqueeze(0),
                dim=-1
            ) / args.temp2
            fmri2fmri_similarity_matrix = torch.exp(fmri2fmri_similarity_matrix)

            pos_sim_index = torch.arange(batch_size * trial_num, device=device).view(batch_size, trial_num).unsqueeze(1) \
                .expand(batch_size, trial_num, trial_num).contiguous().view(batch_size * trial_num, trial_num)
            pos_sim = torch.gather(input=fmri2fmri_similarity_matrix, dim=1, index=pos_sim_index)
            fmri2fmri_cont_loss = torch.mean(-torch.div(pos_sim.sum(dim=1), fmri2fmri_similarity_matrix.sum(dim=1)).log())

            cont_loss = fmri2img_cont_loss * args.lambda1 + fmri2fmri_cont_loss * args.lambda2

            mse_loss = torch.nn.MSELoss()(fmri_fea, img_fea.unsqueeze(1).expand(batch_size, trial_num, dim_fea)).log()
            total_loss = cont_loss + mse_loss * (1 - args.lambda1 - args.lambda2)

            if (step + 1) % (len(dataloader) // 3) == 0:
                cur_time = datetime.datetime.now(dateutil.tz.tzlocal()).strftime('%Y_%m_%d %H:%M:%S')
                print('{} epoch {} | {}/{} step | loss {:.4f} [mse {:.4f}, cont {:.4f} {:.4f}]'.format(cur_time,
                                                    epoch + 1, (step + 1) * batch_size, len(dataloader) * batch_size,
                                                    total_loss, mse_loss, fmri2img_cont_loss, fmri2fmri_cont_loss))

        total_loss.backward()
        optimizer.step()


def evaluate(dataloader, brain_decoder):
    brain_decoder.eval()

    pcc_list = []
    mse_list = []
    sim_list = []
    for step, data in enumerate(dataloader):
        fmri_data, img_fea = data
        # fmri_data:  bs * trial_num * dim_fmri
        # img_fea:    bs * dim_fea

        if args.CUDA:
            fmri_data = fmri_data.to(torch.float32).cuda()
            img_fea = img_fea.to(torch.float32).cuda()
        else:
            fmri_data = fmri_data.to(torch.float32)
            img_fea = img_fea.to(torch.float32)
        fmri_fea = brain_decoder(fmri_data)

        batch_size, trial_num, dim_fea = fmri_fea.shape
        # Pearson correlation coefficient
        for i in range(batch_size):
            for j in range(trial_num):
                pcc, _ = pearsonr(fmri_fea[i, j].detach().cpu().numpy(), img_fea[i].detach().cpu().numpy())
                pcc_list.append(pcc)

        # MSE
        mse_loss = torch.nn.MSELoss()(fmri_fea, img_fea.unsqueeze(1).expand(batch_size, trial_num, dim_fea))
        mse_list.append(mse_loss.detach().cpu().numpy())

        # Similarity
        for i in range(batch_size):
            similarity_matrix = torch.cosine_similarity(
                fmri_fea[i].unsqueeze(1),
                fmri_fea[i].unsqueeze(0),
                dim=-1
            )
            sim = torch.mean(similarity_matrix)
            sim_list.append(sim.detach().cpu().numpy())

    pcc_avg = np.mean(pcc_list)
    mse_avg = np.mean(mse_list)
    sim_avg = np.mean(sim_list)
    return pcc_avg, mse_avg, sim_avg


if __name__ == '__main__':
    args = parse_args()

    if args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    random.seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if args.CUDA:
        torch.cuda.manual_seed_all(args.manualSeed)
        torch.cuda.set_device(args.gpu_id)
        cudnn.benchmark = True
    print(args)

    ##########################################################################
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    output_dir = os.path.join(args.model_dir, timestamp)
    makedir_ifnot(output_dir)

    # Get data loader ##################################################
    dataset_train = BrainDataset(args.fmri_train_path, args.img_fea_train_path)
    dataloader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=args.batch_size, drop_last=True,
        shuffle=True, num_workers=args.num_workers)

    dataset_val = BrainDataset(args.fmri_test_path, args.img_fea_test_path)
    dataloader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=args.batch_size, drop_last=True,
        shuffle=True, num_workers=0)

    # Train ##############################################################
    dim_fmri, dim_img_fea = dataset_train.get_dim()
    print('dataset_train len: %d, dataset_val len: %d' % (len(dataset_train), len(dataset_val)))
    print('dim_fmri: %d, dim_img_fea: %d' % (dim_fmri, dim_img_fea))
    print('-' * 100)
    brain_decoder = BrainDecoder(dim_fmri, dim_img_fea)

    if args.CUDA:
        brain_decoder = brain_decoder.cuda()
    para = list(brain_decoder.parameters())

    # optimizer = optim.Adam(para, lr=cfg.TRAIN.ENCODER_LR, betas=(0.5, 0.999))

    try:
        lr = args.learning_rate
        for epoch in range(args.max_epoch):
            optimizer = optim.Adam(para, lr=lr, betas=(0.5, 0.999), weight_decay=args.weight_decay)
            train(dataloader_train, brain_decoder, optimizer, epoch, args)
            print('-' * 60)

            print('evaluating...')
            pcc, mse, sim = evaluate(dataloader_val, brain_decoder)
            cur_time = datetime.datetime.now(dateutil.tz.tzlocal()).strftime('%Y_%m_%d %H:%M:%S')
            print('{} epoch {} finished. | lr {:.5e} | valid pcc {:.5f} mse {:.5f} sim {:.5f}'.
                  format(cur_time, epoch + 1, lr, pcc, mse, sim))

            if lr > args.learning_rate / 10.:
                lr *= 0.99

            if (epoch + 1) % args.snapshot_interval == 0 or (epoch + 1) == args.max_epoch:
                print('Saving model...')
                torch.save(brain_decoder.state_dict(), '%s/brain_decoder_%d.pth' % (output_dir, epoch + 1))
            print('-' * 100)

    except KeyboardInterrupt:
        # At any point you can hit Ctrl + C to break out of training early.
        print('-' * 100)
        print('Exiting from training early')
