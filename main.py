import os
import argparse
import time
from logging import getLogger, StreamHandler, DEBUG, INFO
from apex.amp.compat import filter_attrs
logger = getLogger(__name__)

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from nltk.translate.bleu_score import corpus_bleu
from torchsummary import summary
from torchvision import models
from torch.nn import Sequential
import torch.nn.functional as F
from reformer_pytorch import Reformer, ReformerLM
from apex import amp
#from apex.parallel import DistributedDataParallel as DDP
from reformer_pytorch.generative_tools import TrainingWrapper
from torch.multiprocessing import set_start_method
try:
    set_start_method('spawn')
except RuntimeError:
    pass
import microsoftvision

from models.dataset import ImageHTMLDataSet, collate_fn_transformer, make_datasets
from models.vocab import build_vocab
from models.metrics import error_exact, accuracy_exact
from models.models import Encoder, Decoder


def set_seed(seed) -> None:
    """
    ランダムシードを固定する
    """
    #random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(seed)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_models(args):

    # define models
    #resnet = models.resnet50(pretrained=True)
    resnet = models.resnet18(pretrained=True)

    #resnet = Sequential(*list(resnet.children())[:-4]) # ([b, 512, 28, 28])
    #resnet = Sequential(*list(resnet.children())[:-2], nn.AdaptiveAvgPool2d((2, 2)))
    #resnet = Sequential(*list(resnet.children())[:-2],
    #                    nn.AdaptiveAvgPool2d((4, 4)))
    resnet = Sequential(*list(resnet.children())[:-2],
                        nn.AdaptiveAvgPool3d((args.dim_reformer, 8, 1)))

    # freeze params
    for p in resnet.parameters():
        p.requires_grad = False

    # encoder = Reformer(
    #     dim=args.dim_reformer,
    #     depth=1,
    #     heads=1,
    #     max_seq_len=256  #4096
    # )
    #seq_len_enc = 1
    seq_len_enc = 8
    encoder = Encoder(
        args.dim_reformer * seq_len_enc,
        args.dim_reformer * seq_len_enc,
        mlp_ratio=1,
        drop=0,
    )

    # decoder = ReformerLM(num_tokens=args.vocab_size,
    #                      dim=args.dim_reformer,
    #                      depth=1,
    #                      heads=1,
    #                      max_seq_len=args.seq_len,
    #                      causal=True)
    # pad = args.vocab('__PAD__')
    # #decoder = TrainingWrapper(decoder, ignore_index=pad, pad_value=pad)
    # decoder = TrainingWrapper(decoder, pad_value=pad)
    seq_len_dec = args.seq_len
    decoder = Decoder(
        args.dim_reformer * seq_len_enc,
        args.vocab_size,
        seq_len_dec,
        mlp_ratio=1,
        drop=0,
    )

    # load models
    if args.step_load != 0:
        trained_model_path = os.path.join(
            args.model_path, 'encoder_{}.pkl'.format(args.step_load))
        encoder.load_state_dict(torch.load(trained_model_path))
        logger.info("loading model: {}".format(trained_model_path))

        trained_model_path = os.path.join(
            args.model_path, 'decoder_{}.pkl'.format(args.step_load))
        decoder.load_state_dict(torch.load(trained_model_path))
        logger.info("loading model: {}".format(trained_model_path))

    print("use pretrain: ", args.use_pretrain)
    if args.use_pretrain:
        trained_model_path = os.path.join(args.model_path,
                                          'decoder_pretrain_30000.pkl')
        decoder.load_state_dict(torch.load(trained_model_path))
        logger.info("loading model: {}".format(trained_model_path))
        print("loading model: {}".format(trained_model_path))

    # set device
    encoder.to(args.device)
    decoder.to(args.device)
    if args.resnet_cpu:
        resnet.to("cpu")
    else:
        resnet.to(args.device)

    return encoder, decoder, resnet


def train(batch_size, encoder, decoder, resnet, args):

    #batch_size = args.batch_size // args.gradient_accumulation_steps

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

    losses = AverageMeter()
    loss_min = 1000
    step_global = 0

    # weight for cross entropy
    ce_weight = torch.tensor(args.list_weight)

    while (step_global < args.step_max):
        for (feature, y_in, lengths, indices) in args.dataloader_train:

            with torch.no_grad():
                if not args.resnet_cpu:
                    feature = feature.to(args.device, non_blocking=True)
                visual_emb = resnet(feature)

            # skip last batch
            if visual_emb.shape[0] != batch_size:
                continue

            logger.debug("visual_emb {}".format(visual_emb.shape))
            b, c, h, w = visual_emb.shape
            # nchw to nte
            #visual_emb = visual_emb.view(b, args.dim_reformer,
            #                             h * w).transpose(1, 2)
            visual_emb = visual_emb.view(b, args.dim_reformer * h * w)
            logger.debug("visual_emb {}".format(visual_emb.shape))
            # 空間部分whがsequence次元に来て，channel部分がdim_enc次元に来たほうが良さそう

            if args.resnet_cpu:
                visual_emb = visual_emb.to(args.device, non_blocking=True)

            # run
            #enc_keys = encoder(visual_emb)
            #logger.debug("enc_keys {}".format(enc_keys.shape))
            #logger.debug(y_in.shape)
            #loss = decoder(y_in, return_loss=True,
            #               keys=enc_keys)
            #logger.debug(y_out.shape)

            y_in = y_in.to(args.device, non_blocking=True)
            ce_weight = ce_weight.to(args.device, non_blocking=True)

            logger.debug("y_in {}".format(y_in.shape))

            enc_keys = encoder(visual_emb)
            logits = decoder(enc_keys, is_train=True)
            logger.debug("logits {}".format(logits.shape))
            loss = F.cross_entropy(logits, y_in, weight=ce_weight)

            logger.debug(loss.item())
            losses.update(loss.item())

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            # backward
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            # update weights
            if (step_global + 1) % args.gradient_accumulation_steps == 0:
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

            # logging
            if (step_global + 1) % args.step_log == 0:
                logger.info("steps: [#%d], loss_train: %.4f" %
                            (step_global, losses.avg))
                losses.reset()

            # validation
            if (step_global + 1) % args.step_save == 0:
                #resnet.to("cpu")

                loss_valid = validate(args.dataloader_valid, encoder, decoder,
                                      resnet, args, ce_weight)
                if loss_valid < loss_min:
                    loss_min = loss_valid
                    # save models
                    logger.info('=== saving models at step: {} ==='.format(
                        step_global))
                    torch.save(
                        decoder.state_dict(),
                        os.path.join(args.model_path,
                                     'decoder_%d.pkl' % (step_global + 1)))
                    torch.save(
                        encoder.state_dict(),
                        os.path.join(args.model_path,
                                     'encoder_%d.pkl' % (step_global + 1)))

                #resnet.to(args.device)

            # end training
            if step_global == args.step_max:
                break
            else:
                step_global += 1

    logger.info('done!')


def validate(dataloader, encoder, decoder, resnet, args, ce_weight):
    encoder.eval()
    decoder.eval()
    eval_losses = AverageMeter()

    for step, (feature, y_in, lengths, indices) in enumerate(dataloader):

        with torch.no_grad():
            if not args.resnet_cpu:
                feature = feature.to(args.device, non_blocking=True)
            visual_emb = resnet(feature)

        y_in = y_in.to(args.device, non_blocking=True)
        b, c, h, w = visual_emb.shape
        # nchw to nte
        #visual_emb = visual_emb.view(b, args.dim_reformer,
        #                             h * w).transpose(1, 2)
        visual_emb = visual_emb.view(b, args.dim_reformer * h * w)

        if args.resnet_cpu:
            visual_emb = visual_emb.to(args.device, non_blocking=True)

        with torch.no_grad():
            # run
            enc_keys = encoder(visual_emb)
            logits = decoder(enc_keys, is_train=True)
            loss = F.cross_entropy(logits, y_in, weight=ce_weight)

            eval_losses.update(loss.item())
            logger.debug("Loss: %.4f" % (eval_losses.avg))
    logger.info("loss_valid: %.4f" % (eval_losses.avg))

    encoder.train()
    decoder.train()

    return eval_losses.avg


def predict(dataloader, encoder, decoder, resnet, args):
    encoder.eval()
    decoder.eval()

    tags_pred = []
    tags_gt = []

    pad = args.vocab('__PAD__')
    bgn = args.vocab('__BGN__')
    end = args.vocab('__END__')
    cnt = 0

    for step, (feature, y_in, lengths, indices) in enumerate(tqdm(dataloader)):
        #if step < 271:
        #    continue

        with torch.no_grad():
            if not args.resnet_cpu:
                feature = feature.to(args.device, non_blocking=True)
            visual_emb = resnet(feature)

        y_in = y_in.to('cpu')
        b, c, h, w = visual_emb.shape
        # nchw to nte
        #visual_emb = visual_emb.view(b, args.dim_reformer,
        #                             h * w).transpose(1, 2)
        visual_emb = visual_emb.view(b, args.dim_reformer * h * w)

        if args.resnet_cpu:
            visual_emb = visual_emb.to(args.device, non_blocking=True)

        with torch.no_grad():
            # run
            enc_keys = encoder(visual_emb)
            logits = decoder(enc_keys)
            samples = torch.argmax(logits, dim=1)

            logger.debug("generated sentence: {}".format(samples))
            logger.debug("ground truth: {}".format(y_in))

            samples = samples.to('cpu').detach().numpy().copy()
            str_pred = ""
            str_gt = ""
            for i, sample in enumerate(samples):
                idx = int(indices[i].to('cpu').detach().numpy().copy().item())
                name, _ = os.path.splitext(
                    os.path.basename(dataloader.dataset.paths_image[idx]))

                # preserve prediction
                #tags = [args.vocab.idx2word[str(bgn)]] + [
                #    args.vocab.idx2word[str(int(x))]
                #    for x in sample if not x == pad
                #]
                tags = [args.vocab.idx2word[str(bgn)]]
                for x in sample:
                    if x == pad:
                        continue
                    tags.append(args.vocab.idx2word[str(int(x))])
                    if x == end:
                        break
                tags_pred.append(tags)

                # save file
                str_pred = "\n".join(tags)
                #path = os.path.join(args.out_dir_pred, str(cnt) + "_pred.html")
                path = os.path.join(args.out_dir_pred,
                                    str(name) + "_pred.html")
                with open(path, "w") as f:
                    f.write(str_pred)

                # preserve ground truth
                gt = [
                    args.vocab.idx2word[str(int(x))] for x in y_in[i]
                    if not x == pad
                ]
                tags_gt.append([gt])

                # save file
                str_gt = "\n".join(gt)
                #path = os.path.join(args.out_dir_gt, str(cnt) + "_gt.html")
                path = os.path.join(args.out_dir_gt, str(name) + "_gt.html")
                with open(path, "w") as f:
                    f.write(str_gt)

                cnt += 1

    return tags_pred, tags_gt


def test(encoder, decoder, resnet, args):

    tags_pred, tags_gt = predict(args.dataloader_test, encoder, decoder,
                                 resnet, args)

    # calc scores
    bleu = corpus_bleu(tags_gt, tags_pred)
    err = error_exact(tags_gt, tags_pred)
    acc = accuracy_exact(tags_gt, tags_pred)

    logger.info("bleu score: {}".format(bleu))
    logger.info("error : {}".format(err))
    logger.info("accuracy: {}".format(acc))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name",
                        default="018_transformer_flat_p2c")
    parser.add_argument("--data_name", default="015_flat_seq_pix2code")
    parser.add_argument("--ckpt_name", default="ckpt")
    parser.add_argument("--mode", default="train")
    parser.add_argument("--step_load", type=int, default=0)
    parser.add_argument("--step_max", type=int, default=100000)
    parser.add_argument("--step_log", type=int, default=10000)
    parser.add_argument("--step_save", type=int, default=10000)

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
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--log_level", type=str, default="DEBUG")
    parser.add_argument('--use_pretrain', action='store_true')
    parser.add_argument('--resnet_cpu', action='store_true')

    args = parser.parse_args()

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
    args.out_dir_pred = exp_root + "/test/pred"
    args.out_dir_gt = exp_root + "/test/gt"

    # data csv
    args.data_path_csv_train = exp_root + "/dataset/train.csv"
    args.data_path_csv_valid = exp_root + "/dataset/valid.csv"
    args.data_path_csv_test = exp_root + "/dataset/test.csv"

    # checkpoint
    args.model_path = exp_root + "/" + args.ckpt_name

    if not os.path.exists(exp_root):
        os.mkdir(exp_root)
    if not os.path.exists(args.model_path):
        os.mkdir(args.model_path)
    if not os.path.exists(args.out_dir_pred):
        os.makedirs(args.out_dir_pred)
    if not os.path.exists(args.out_dir_gt):
        os.makedirs(args.out_dir_gt)

    # Hyperparams
    args.learning_rate = 0.001
    args.seq_len = 2048
    args.dim_reformer = 32

    # Other params
    args.shuffle_train = True
    args.shuffle_test = False
    args.max_sample = args.seq_len  # for predictions
    args.crop_size = 256

    # vocabulary
    args.path_vocab_txt = exp_root + "/vocab.txt"
    args.path_vocab_w2i = exp_root + '/w2i.json'
    args.path_vocab_i2w = exp_root + '/i2w.json'

    set_seed(42)

    # dataset
    batch_size = make_datasets(args)

    # model
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder, decoder, resnet = get_models(args)

    # log level
    log_level = args.log_level
    handler = StreamHandler()
    handler.setLevel(log_level)
    logger.setLevel(log_level)
    logger.addHandler(handler)
    logger.propagate = False

    start = time.time()
    logger.info("mode: {}".format(args.mode))
    logger.info("num_workers: {}".format(args.num_workers))
    # start training
    if args.mode == "train":
        train(batch_size, encoder, decoder, resnet, args)
    else:
        test(encoder, decoder, resnet, args)
    elapsed_time = time.time() - start
    logger.info("elapsed_time:{0}".format(elapsed_time / 3600) + "[h]")
