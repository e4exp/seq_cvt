import os
import argparse
import time
from logging import getLogger, StreamHandler, DEBUG, INFO
logger = getLogger(__name__)

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision
from nltk.translate.bleu_score import corpus_bleu
from torchsummary import summary
from torchvision import models
from torch.nn import Sequential
import torch.nn.functional as F
from reformer_pytorch import Reformer, ReformerLM
from apex import amp
from reformer_pytorch.generative_tools import TrainingWrapper
#import microsoftvision
from torch.utils.tensorboard import SummaryWriter

from models.dataset import make_datasets
from models.metrics import error_exact, accuracy_exact
from models.scheduler import WarmupLinearSchedule, WarmupCosineSchedule


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
    resnet = Sequential(*list(resnet.children())[:-2],
                        nn.AdaptiveAvgPool3d((args.dim_reformer, 64, 8)))

    if args.resnet_cpu:
        resnet.to("cpu")
    else:
        resnet.to(args.device)

    # parameters
    params = list(resnet.parameters())
    opt_resnet = torch.optim.Adam(params, lr=args.learning_rate)
    # set precision
    if args.fp16:
        resnet, opt_resnet = amp.initialize(models=resnet,
                                            optimizers=opt_resnet,
                                            opt_level=args.fp16_opt_level)
        amp._amp_state.loss_scalers[0]._loss_scale = 2**20  #?

    # encoder
    encoder = Reformer(
        dim=args.dim_reformer,
        depth=1,
        heads=1,
        max_seq_len=256,
        weight_tie=False,  # default=False
        use_full_attn=True,
        #ff_dropout=0.1,
        #post_attn_dropout=0.1,
        #layer_dropout=0.1,
    )

    encoder.to(args.device)

    # # parameters
    params = list(encoder.parameters())
    opt_encoder = torch.optim.Adam(params, lr=args.learning_rate)
    # set precision
    if args.fp16:
        encoder, opt_encoder = amp.initialize(models=encoder,
                                              optimizers=opt_encoder,
                                              opt_level=args.fp16_opt_level)
        amp._amp_state.loss_scalers[0]._loss_scale = 2**20

    # decoder
    decoder = ReformerLM(
        num_tokens=args.vocab_size,
        dim=args.dim_reformer,
        depth=2,
        heads=4,
        max_seq_len=args.seq_len,
        weight_tie=False,
        weight_tie_embedding=False,
        use_full_attn=True,
        #ff_dropout=0.1,
        #post_attn_dropout=0.1,
        #layer_dropout=0.1,
        causal=True)
    pad = args.vocab('__PAD__')
    decoder = TrainingWrapper(decoder, pad_value=pad)

    decoder.to(args.device)

    # parameters
    params = list(decoder.parameters())
    opt_decoder = torch.optim.Adam(params, lr=args.learning_rate)
    # set precision
    if args.fp16:
        decoder, opt_decoder = amp.initialize(models=decoder,
                                              optimizers=opt_decoder,
                                              opt_level=args.fp16_opt_level)
        amp._amp_state.loss_scalers[0]._loss_scale = 2**20

    # load models
    if args.epoch_load != 0:
        # encoder
        trained_model_path = os.path.join(
            args.model_path, 'encoder_{}.pkl'.format(args.epoch_load))
        encoder.load_state_dict(torch.load(trained_model_path))
        logger.info("loading model: {}".format(trained_model_path))

        # decoder
        trained_model_path = os.path.join(
            args.model_path, 'decoder_{}.pkl'.format(args.epoch_load))
        decoder.load_state_dict(torch.load(trained_model_path))
        logger.info("loading model: {}".format(trained_model_path))

        # resnet
        trained_model_path = os.path.join(
            args.model_path, 'resnet_{}.pkl'.format(args.epoch_load))
        resnet.load_state_dict(torch.load(trained_model_path))
        logger.info("loading model: {}".format(trained_model_path))

    print("use pretrain: ", args.use_pretrain)
    if args.use_pretrain:
        trained_model_path = os.path.join(args.model_path,
                                          'decoder_pretrain_10000.pkl')
        decoder.load_state_dict(torch.load(trained_model_path))
        logger.info("loading model: {}".format(trained_model_path))
        print("loading model: {}".format(trained_model_path))

    args.opt_encoder = opt_encoder
    args.opt_decoder = opt_decoder
    args.opt_resnet = opt_resnet

    return encoder, decoder, resnet
    #return decoder, resnet


def get_schedulers(args, opt_enc, opt_dec, opt_res, step_max):
    #def get_schedulers(args, opt_dec, opt_res, step_max):

    if args.decay_type == "cosine":
        scheduler_enc = WarmupCosineSchedule(opt_enc,
                                             warmup_steps=args.warmup_steps,
                                             t_total=step_max)
        scheduler_dec = WarmupCosineSchedule(opt_dec,
                                             warmup_steps=args.warmup_steps,
                                             t_total=step_max)
        scheduler_res = WarmupCosineSchedule(opt_res,
                                             warmup_steps=args.warmup_steps,
                                             t_total=step_max)
    else:
        scheduler_enc = WarmupLinearSchedule(opt_enc,
                                             warmup_steps=args.warmup_steps,
                                             t_total=step_max)
        scheduler_dec = WarmupLinearSchedule(opt_dec,
                                             warmup_steps=args.warmup_steps,
                                             t_total=step_max)
        scheduler_res = WarmupLinearSchedule(opt_res,
                                             warmup_steps=args.warmup_steps,
                                             t_total=step_max)

    return scheduler_enc, scheduler_dec, scheduler_res
    #return scheduler_dec, scheduler_res


def train(batch_size, encoder, decoder, resnet, args):
    #def train(batch_size, decoder, resnet, args):

    #batch_size = args.batch_size // args.gradient_accumulation_steps
    # optimizers
    opt_enc = args.opt_encoder
    opt_dec = args.opt_decoder
    opt_res = args.opt_resnet

    # schedulers
    step_scheduler = args.step_max // args.gradient_accumulation_steps
    scheduler_enc, scheduler_dec, scheduler_res, = get_schedulers(
        args, opt_enc, opt_dec, opt_res, step_scheduler)
    # scheduler_dec, scheduler_res, = get_schedulers(args, opt_dec, opt_res,
    #                                                step_scheduler)

    encoder.train()
    decoder.train()
    #resnet.eval()
    resnet.train()

    # summary
    logger.debug("encoder: ")
    summary(encoder)
    logger.debug("decoder: ")
    summary(decoder)

    #losses = AverageMeter()
    losses_epoch = AverageMeter()
    loss_min = 1000
    epoch_best = 0
    #step_global = args.step_load
    step_global = 0
    epoch_global = args.epoch_load
    #sample_global = 0

    # weight for cross entropy
    ce_weight = torch.tensor(args.list_weight).to(args.device,
                                                  non_blocking=True)

    while (epoch_global < args.epoch_max):
        #while (sample_global < args.sample_max):
        for (feature, y_in, lengths, indices) in args.dataloader_train:

            img_grid = torchvision.utils.make_grid(feature, nrow=8)
            args.writer.add_image('images_train',
                                  img_tensor=img_grid,
                                  global_step=step_global)

            if not args.resnet_cpu:
                feature = feature.to(args.device, non_blocking=True)
            visual_emb = resnet(feature)

            # skip last batch
            if visual_emb.shape[0] != batch_size:
                continue

            logger.debug("visual_emb {}".format(visual_emb.shape))
            b, c, h, w = visual_emb.shape
            # nchw to nte
            visual_emb = visual_emb.view(b, c, h * w).transpose(1, 2)
            #visual_emb = visual_emb.view(b, args.dim_reformer * h * w)
            logger.debug("visual_emb {}".format(visual_emb.shape))
            # 空間部分whがsequence次元に来て，channel部分がdim_enc次元に来たほうが良さそう

            if args.resnet_cpu:
                visual_emb = visual_emb.to(args.device, non_blocking=True)

            y_in = y_in.to(args.device, non_blocking=True)

            # run
            enc_keys = encoder(visual_emb)
            # logger.debug("enc_keys {}".format(enc_keys.shape))
            # logger.debug(y_in.shape)
            _, loss = decoder(y_in, return_loss=True, keys=enc_keys)

            logger.debug(loss.item())
            #losses.update(loss.item())
            losses_epoch.update(loss.item())

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            # backward
            if args.fp16:
                with amp.scale_loss(loss, opt_enc) as scaled_loss:
                    scaled_loss.backward(retain_graph=True)
                with amp.scale_loss(loss, opt_dec) as scaled_loss:
                    scaled_loss.backward(retain_graph=True)
                with amp.scale_loss(loss, opt_res) as scaled_loss:
                    scaled_loss.backward()

            else:
                loss.backward()

            args.writer.add_scalar("train/loss",
                                   scalar_value=loss.item(),
                                   global_step=step_global)

            # update weights
            if (step_global + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(opt_enc),
                                                   args.max_grad_norm)
                    torch.nn.utils.clip_grad_norm_(amp.master_params(opt_dec),
                                                   args.max_grad_norm)
                    torch.nn.utils.clip_grad_norm_(amp.master_params(opt_res),
                                                   args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(encoder.parameters(),
                                                   args.max_grad_norm)
                    torch.nn.utils.clip_grad_norm_(decoder.parameters(),
                                                   args.max_grad_norm)
                    torch.nn.utils.clip_grad_norm_(resnet.parameters(),
                                                   args.max_grad_norm)

                opt_enc.step()
                opt_enc.zero_grad()
                opt_dec.step()
                opt_dec.zero_grad()
                opt_res.step()
                opt_res.zero_grad()

                scheduler_enc.step()
                scheduler_dec.step()
                scheduler_res.step()

            step_global += 1

        # logging
        if (epoch_global + 1) % args.epoch_log == 0:
            logger.info("epochs: [#%d], loss_train: %.4f" %
                        (epoch_global, losses_epoch.avg))
            losses_epoch.reset()

        # validation
        if (epoch_global + 1) % args.epoch_valid == 0:
            loss_valid = validate(args.dataloader_valid, encoder, decoder,
                                  resnet, args, epoch_global, ce_weight)
            #loss_valid = validate(args.dataloader_valid, decoder, resnet,
            #                      args, step_global, ce_weight)
            if loss_valid < loss_min:
                loss_min = loss_valid
                epoch_best = epoch_global + 1
                #save_models(args, epoch_global + 1, encoder, decoder, resnet)
                #save_models(args, step_global + 1, decoder, resnet)

        # save
        if (epoch_global + 1) % args.epoch_save == 0:
            save_models(args, epoch_global + 1, encoder, decoder, resnet)
            #save_models(args, step_global + 1, decoder, resnet)

        epoch_global += 1

    # save final model
    # loss_valid = validate(args.dataloader_valid, encoder, decoder, resnet,
    #                       args, step_global, ce_weight)
    # if loss_valid < loss_min:
    #     epoch_best = epoch_global + 1
    # save_models(args, epoch_global + 1, encoder, decoder, resnet)

    # log final
    # logger.info("epoch: [#%d], loss_train: %.4f" %
    #             (epoch_global, losses_epoch.avg))
    # losses_epoch.reset()

    args.writer.close()
    logger.info('done!')
    logger.info("best model was in {}".format(epoch_best))


def save_models(args, step, encoder, decoder, resnet):
    #def save_models(args, step, decoder, resnet):

    # save models
    #logger.info('=== saving models at step: {} ==='.format(step))
    logger.info('=== saving models at epoch: {} ==='.format(step))
    torch.save(decoder.state_dict(),
               os.path.join(args.model_path, 'decoder_%d.pkl' % (step)))
    torch.save(encoder.state_dict(),
               os.path.join(args.model_path, 'encoder_%d.pkl' % (step)))
    torch.save(resnet.state_dict(),
               os.path.join(args.model_path, 'resnet_%d.pkl' % (step)))


def validate(dataloader,
             encoder,
             decoder,
             resnet,
             args,
             epoch,
             ce_weight=None):
    #def validate(dataloader, decoder, resnet, args, step, ce_weight=None):
    encoder.eval()
    decoder.eval()
    resnet.eval()
    eval_losses = AverageMeter()

    for i, (feature, y_in, lengths, indices) in enumerate(dataloader):

        with torch.no_grad():
            if not args.resnet_cpu:
                feature = feature.to(args.device, non_blocking=True)
            visual_emb = resnet(feature)

        y_in = y_in.to(args.device, non_blocking=True)
        b, c, h, w = visual_emb.shape
        # nchw to nte
        visual_emb = visual_emb.view(b, c, h * w).transpose(1, 2)
        #visual_emb = visual_emb.view(b, args.dim_reformer * h * w)

        if args.resnet_cpu:
            visual_emb = visual_emb.to(args.device, non_blocking=True)

        with torch.no_grad():
            # run
            enc_keys = encoder(visual_emb)
            _, loss = decoder(y_in, return_loss=True, keys=enc_keys)

            eval_losses.update(loss.item())
            logger.debug("Loss: %.4f" % (eval_losses.avg))
    logger.info("loss_valid: %.4f" % (eval_losses.avg))

    img_grid = torchvision.utils.make_grid(feature[:8], nrow=2)
    args.writer.add_image('images_val', img_tensor=img_grid, global_step=epoch)
    args.writer.add_scalar("train/val",
                           scalar_value=eval_losses.avg,
                           global_step=epoch)

    encoder.train()
    decoder.train()
    resnet.train()

    return eval_losses.avg


def predict(dataloader, encoder, decoder, resnet, args):
    #def predict(dataloader, decoder, resnet, args):
    encoder.eval()
    decoder.eval()

    tags_pred = []
    tags_gt = []

    pad = args.vocab('__PAD__')
    bgn = args.vocab('__BGN__')
    #bgn = args.vocab('<html>')
    end = args.vocab('__END__')
    cnt = 0

    for step, (feature, y_in, lengths, indices) in enumerate(tqdm(dataloader)):
        #if step < 247:
        #    continue

        with torch.no_grad():
            if not args.resnet_cpu:
                feature = feature.to(args.device, non_blocking=True)
            visual_emb = resnet(feature)

        y_in = y_in.to('cpu')
        b, c, h, w = visual_emb.shape
        # nchw to nte
        visual_emb = visual_emb.view(b, c, h * w).transpose(1, 2)
        #visual_emb = visual_emb.view(b, args.dim_reformer * h * w)

        if args.resnet_cpu:
            visual_emb = visual_emb.to(args.device, non_blocking=True)

        initial = torch.tensor([[bgn]]).long().repeat([b, 1
                                                       ]).to(args.device,
                                                             non_blocking=True)
        with torch.no_grad():
            # run
            enc_keys = encoder(visual_emb)
            samples = decoder.generate(
                initial,
                args.seq_len,
                temperature=1.,
                filter_thres=0.9,
                eos_token=end,
                keys=enc_keys,
            )  # assume end token is 1, or omit and it will sample up to 100

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
                tags = [
                    args.vocab.idx2word[str(int(x))] for x in sample
                    if not x == pad
                ]
                tags.insert(0, args.vocab.idx2word[str(bgn)])
                # tags = [args.vocab.idx2word[str(bgn)]]
                # for x in sample:
                #     if x == pad:
                #         continue
                #     tags.append(args.vocab.idx2word[str(int(x))])
                #     if x == end:
                #         break
                tags_pred.append(tags)

                # save file
                str_pred = "\n".join(tags)
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
                path = os.path.join(args.out_dir_gt, str(name) + "_gt.html")
                with open(path, "w") as f:
                    f.write(str_gt)

    return tags_pred, tags_gt


def test(encoder, decoder, resnet, args):
    #def test(decoder, resnet, args):

    tags_pred, tags_gt = predict(args.dataloader_test, encoder, decoder,
                                 resnet, args)
    #tags_pred, tags_gt = predict(args.dataloader_test, decoder, resnet, args)

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

    parser.add_argument("--epoch_load", type=int, default=0)
    parser.add_argument("--epoch_max", type=int, default=10)
    parser.add_argument("--epoch_log", type=int, default=1)
    parser.add_argument("--epoch_save", type=int, default=10)
    parser.add_argument("--epoch_valid", type=int, default=5)

    parser.add_argument("--step_load", type=int, default=0)
    parser.add_argument("--step_max", type=int, default=100000)
    parser.add_argument("--step_log", type=int, default=1000)
    parser.add_argument("--step_save", type=int, default=10000)
    parser.add_argument("--step_valid", type=int, default=5000)

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
    parser.add_argument(
        "--warmup_steps",
        default=500,
        type=int,
        help="Step of training to perform learning rate warmup for.")
    parser.add_argument("--decay_type",
                        choices=["cosine", "linear"],
                        default="cosine",
                        help="How to decay the learning rate.")

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

    # tensorboard
    args.log_dir = exp_root + "/" + "logs"

    if not os.path.exists(exp_root):
        os.mkdir(exp_root)
    if not os.path.exists(args.model_path):
        os.mkdir(args.model_path)
    if not os.path.exists(args.out_dir_pred):
        os.makedirs(args.out_dir_pred)
    if not os.path.exists(args.out_dir_gt):
        os.makedirs(args.out_dir_gt)
    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)

    # Hyperparams
    args.learning_rate = 0.001
    args.seq_len = 2048
    args.dim_reformer = 512

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

    # tensorboard
    args.writer = SummaryWriter(log_dir=args.log_dir)

    # log level
    log_level = args.log_level
    handler = StreamHandler()
    handler.setLevel(log_level)
    logger.setLevel(log_level)
    logger.addHandler(handler)
    logger.propagate = False

    # dataset
    batch_size = make_datasets(args)

    # model
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder, decoder, resnet = get_models(args)
    #decoder, resnet = get_models(args)

    start = time.time()
    logger.info("mode: {}".format(args.mode))
    logger.info("num_workers: {}".format(args.num_workers))
    # start training
    if args.mode == "train":
        train(batch_size, encoder, decoder, resnet, args)
        #train(batch_size, decoder, resnet, args)
    else:
        test(encoder, decoder, resnet, args)
        #test(decoder, resnet, args)
    elapsed_time = time.time() - start
    logger.info("elapsed_time:{0}".format(elapsed_time / 3600) + "[h]")
