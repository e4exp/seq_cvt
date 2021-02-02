import os
import argparse
import time
import glob
import math
from logging import getLogger, StreamHandler, DEBUG, INFO
log_level = DEBUG
logger = getLogger(__name__)
handler = StreamHandler()
handler.setLevel(log_level)
logger.setLevel(log_level)
logger.addHandler(handler)
logger.propagate = False

import numpy as np
from torch._C import device
from tqdm import tqdm
from PIL import Image
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from nltk.translate.bleu_score import corpus_bleu
from torchsummary import summary
from torchtext import data
from torchvision import models
from torch.nn import Sequential
from reformer_pytorch import Reformer, ReformerLM
#from apex import amp
#from apex.parallel import DistributedDataParallel as DDP
from reformer_pytorch.generative_tools import TrainingWrapper

from torch.multiprocessing import set_start_method
try:
    set_start_method('spawn')
except RuntimeError:
    pass

from models.dataset import ImageHTMLDataSet, collate_fn_transformer
from models.vocab import build_vocab


def get_models(args):

    # define models
    #resnet = models.resnet50(pretrained=True)
    resnet = models.resnet18(pretrained=True)
    #resnet = Sequential(*list(resnet.children())[:-4]) # ([b, 512, 28, 28])
    #resnet = Sequential(*list(resnet.children())[:-2], nn.AdaptiveAvgPool2d((2, 2)))
    resnet = Sequential(*list(resnet.children())[:-2],
                        nn.AdaptiveAvgPool2d((4, 4)))

    encoder = Reformer(
        dim=args.dim_reformer,
        depth=1,
        heads=1,
        max_seq_len=256  #4096
    )

    decoder = ReformerLM(num_tokens=args.vocab_size,
                         dim=args.dim_reformer,
                         depth=1,
                         heads=1,
                         max_seq_len=args.seq_len,
                         causal=True)

    # load models
    if args.step_load != 0:
        trained_model_path = os.path.join(
            args.model_path, 'encoder_{}.pkl'.format(args.step_load))
        encoder.load_state_dict(torch.load(trained_model_path))
        logger.debug("loading model: {}".format(trained_model_path))

        trained_model_path = os.path.join(
            args.model_path, 'decoder_{}.pkl'.format(args.step_load))
        decoder.load_state_dict(torch.load(trained_model_path))
        logger.debug("loading model: {}".format(trained_model_path))

    # set device
    encoder.to(args.device)
    decoder.to(args.device)
    resnet.to(args.device)

    return encoder, decoder, resnet


def train(encoder, decoder, resnet, args):

    batch_size = args.batch_size // args.gradient_accumulation_steps

    # loss
    criterion_ce = nn.CrossEntropyLoss()

    # parameters
    params = list(decoder.parameters()) + list(encoder.parameters())
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)

    # set precision
    if args.fp16:
        models_fp, optimizer = amp.initialize(models=[encoder, decoder],
                                              optimizers=optimizer,
                                              opt_level=args.fp16_opt_level)
        amp._amp_state.loss_scalers[0]._loss_scale = 2**20
        encoder, decoder = models_fp

    encoder.train()
    decoder.train()
    resnet.eval()

    # summary
    logger.debug("encoder: ")
    summary(encoder)
    logger.debug("decoder: ")
    summary(decoder)

    # dataset
    transform = transforms.Compose([
        transforms.RandomResizedCrop(args.crop_size,
                                     scale=(1.0, 1.0),
                                     ratio=(1.0, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    dataset = ImageHTMLDataSet(args.data_dir_img, args.data_dir_html,
                               args.vocab, transform, resnet, args.device)
    dataloader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=args.shuffle_train,
                            num_workers=args.num_workers,
                            collate_fn=collate_fn_transformer)

    for epoch in range(args.step_load, args.num_epochs + args.step_load):
        losses_t = []
        for step, (visual_emb, y_in, lengths) in enumerate(dataloader):

            # skip last batch
            if visual_emb.shape[0] != batch_size:
                continue
            y_in = y_in.to(args.device)

            #torch.cuda.empty_cache()
            b, s, c, h, w = visual_emb.shape
            # nchw to nte
            #visual_emb = visual_emb.view(b, c, s * h * w).transpose(1, 2) # nchw to nte
            visual_emb = visual_emb.view(b, args.dim_reformer,
                                         c // args.dim_reformer * s * h *
                                         w).transpose(1, 2)
            #logger.debug("visual_emb {}".format(visual_emb.shape))

            # run
            enc_keys = encoder(visual_emb)
            #logger.debug(enc_keys.shape)
            #logger.debug(y_in.shape)
            y_out = decoder(y_in, keys=enc_keys)  # (batch, seq, vocab)
            #logger.debug(y_out.shape)

            loss = criterion_ce(
                y_out.reshape(batch_size * args.seq_len, args.vocab_size),
                y_in.reshape(batch_size * args.seq_len).long())
            logger.debug(loss)

            #losses_t.append(loss.item() / batch_size)
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                #losses.update(loss.item() * args.gradient_accumulation_steps)
                losses_t.append(loss.item() / batch_size *
                                args.gradient_accumulation_steps)
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(
                        amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(encoder.parameters(),
                                                   args.max_grad_norm)
                    torch.nn.utils.clip_grad_norm_(decoder.parameters(),
                                                   args.max_grad_norm)

                optimizer.step()
                optimizer.zero_grad()
                #logger.debug("zero_grad")

        if epoch % args.step_log == 0:
            logger.info("Epoch [#%d], Loss_t: %.4f" %
                        (epoch, np.mean(losses_t)))

        if (epoch + 1) % args.step_save == 0:
            # save models
            logger.info('!!! saving models at epoch: ' + str(epoch))
            torch.save(
                decoder.state_dict(),
                os.path.join(args.model_path, 'decoder_%d.pkl' % (epoch + 1)))
            torch.save(
                encoder.state_dict(),
                os.path.join(args.model_path, 'encoder_%d.pkl' % (epoch + 1)))

    logger.info('done!')


def validate(dataloader, encoder, decoder, resnet, criterion, args):
    encoder.eval()
    decoder.eval()
    resnet.eval()

    for step, (visual_emb, y_in, lengths) in enumerate(dataloader):

        y_in = y_in.to(args.device)
        b, s, c, h, w = visual_emb.shape
        # nchw to nte
        visual_emb = visual_emb.view(b, args.dim_reformer,
                                     c // args.dim_reformer * s * h *
                                     w).transpose(1, 2)

        # run
        enc_keys = encoder(visual_emb)
        y_out = decoder(y_in, keys=enc_keys)  # (batch, seq, vocab)

        loss = criterion(
            y_out.reshape(args.batch_size_val * args.seq_len, args.vocab_size),
            y_in.reshape(args.batch_size_val * args.seq_len).long())
        logger.debug(loss)
        logger.info("Loss_t: %.4f" % (np.mean(loss)))

    encoder.train()
    decoder.train()
    resnet.eval()


def test(encoder, decoder, resnet, args):
    # loss
    criterion_ce = nn.CrossEntropyLoss()

    # dataset
    transform = transforms.Compose([
        transforms.RandomResizedCrop(args.crop_size,
                                     scale=(1.0, 1.0),
                                     ratio=(1.0, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    dataset = ImageHTMLDataSet(args.data_dir_img_test, args.data_dir_html_test,
                               args.vocab, transform, resnet, args.device)
    dataloader = DataLoader(dataset=dataset,
                            batch_size=args.batch_size_val,
                            shuffle=args.shuffle_train,
                            num_workers=args.num_workers,
                            collate_fn=collate_fn_transformer)

    validate(dataloader, encoder, decoder, resnet, criterion_ce, args)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name",
                        default="018_transformer_flat_p2c")
    parser.add_argument("--data_name", default="015_flat_seq_pix2code")
    parser.add_argument("--ckpt_name", default="ckpt")
    parser.add_argument("--mode", default="train")
    parser.add_argument("--step_load", type=int, default=0)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument(
        '--fp16',
        action='store_true',
        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument(
        '--fp16_opt_level',
        type=str,
        default='O2',
        help=
        "For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument(
        '--loss_scale',
        type=float,
        default=0,
        help=
        "Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
        "0 (default value): dynamic loss scaling.\n"
        "Positive power of 2: static loss scaling value.\n")
    parser.add_argument(
        '--gradient_accumulation_steps',
        type=int,
        default=1,
        help=
        "Number of updates steps to accumulate before performing a backward/update pass."
    )
    parser.add_argument("--max_grad_norm",
                        default=1.0,
                        type=float,
                        help="Max gradient norm.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--batch_size_val", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)

    experiment_name = "026_reformer"
    #data_name = "015_flat_seq_pix2code"
    data_name = "014_flat_seq"
    ckpt_name = "ckpt"
    g_steps = 8
    mode = "train"
    #mode="test"
    opt_level = "O2"

    args = parser.parse_args([
                            "--experiment_name", experiment_name, \
                            "--data_name", data_name, \
                            "--ckpt_name", ckpt_name, \
                            "--mode", mode, \
                            #"--fp16", \
                            "--fp16_opt_level", opt_level, \
                            "--gradient_accumulation_steps", str(g_steps), \
                            ])

    # Paths
    root = "../drnn/experiments"
    exp_root = root + "/" + args.experiment_name
    data_root = root + "/" + args.data_name
    data_dir = data_root + "/dataset"

    # train
    args.data_dir_img = data_dir + "/train/img"
    args.data_dir_html = data_dir + "/train/html"

    # test
    args.data_dir_img_test = data_dir + "/test/img"
    args.data_dir_html_test = data_dir + "/test/html"

    # checkpoint
    args.model_path = exp_root + "/" + args.ckpt_name

    if not os.path.exists(exp_root):
        os.mkdir(exp_root)
    if not os.path.exists(args.model_path):
        os.mkdir(args.model_path)

    # Hyperparams
    args.learning_rate = 0.001
    args.seq_len = 4096
    args.dim_reformer = 256

    # Other params
    args.shuffle_train = True
    args.shuffle_test = False
    args.max_sample = args.seq_len  # for predictions

    # Logging Variables
    args.step_save = 20
    args.step_log = 1

    args.crop_size = 256

    # vocabulary
    vocab_dir = args.model_path
    args.path_vocab_txt = exp_root + "/vocab.txt"
    args.path_vocab_w2i = exp_root + '/w2i.json'
    args.path_vocab_i2w = exp_root + '/i2w.json'

    # vocab
    args.vocab = build_vocab(args.path_vocab_txt, args.path_vocab_w2i,
                             args.path_vocab_i2w)
    args.vocab_size = len(args.vocab)

    # model
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder, decoder, resnet = get_models(args)

    start = time.time()
    # start training
    if args.mode == "train":
        train(encoder, decoder, resnet, args)
    else:
        test(encoder, decoder, resnet, args)
    elapsed_time = time.time() - start
    logger.info("elapsed_time:{0}".format(elapsed_time / 3600) + "[h]")
