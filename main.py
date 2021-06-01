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
import torchvision
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
#from torch.multiprocessing import set_start_method
#try:
#    set_start_method('spawn')
#except RuntimeError:
#    pass
import microsoftvision
from torch.utils.tensorboard import SummaryWriter

from models.dataset import ImageHTMLDataSet, collate_fn_transformer, make_datasets
from models.vocab import build_vocab
from models.metrics import error_exact, accuracy_exact
from models.models import Discriminator
from models.utils import gradient_penalty


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
    #resnet = Sequential(*list(resnet.children())[:-2],
    #                    nn.AdaptiveAvgPool3d((args.dim_reformer, 8, 2)))
    resnet = Sequential(*list(resnet.children())[:-2],
                        nn.AdaptiveAvgPool3d((args.dim_reformer, 64, 8)))

    # freeze params
    for p in resnet.parameters():
        p.requires_grad = False

    encoder = Reformer(
        dim=args.dim_reformer,
        depth=1,
        heads=1,
        max_seq_len=256,  # <- this is dummy param
        weight_tie=False,  # default=False
    )

    decoder = ReformerLM(num_tokens=args.vocab_size,
                         dim=args.dim_reformer,
                         depth=1,
                         heads=1,
                         max_seq_len=args.seq_len,
                         weight_tie=False,
                         causal=True)
    pad = args.vocab('__PAD__')
    #decoder = TrainingWrapper(decoder, ignore_index=pad, pad_value=pad)
    decoder = TrainingWrapper(decoder, pad_value=pad)

    D = Discriminator(args.vocab_size, args.seq_len)

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
    D.to(args.device)
    if args.resnet_cpu:
        resnet.to("cpu")
    else:
        resnet.to(args.device)

    return encoder, decoder, D, resnet


def train(batch_size, encoder, decoder, D, resnet, args):

    #batch_size = args.batch_size // args.gradient_accumulation_steps
    # tensorboard

    ims, _, _, _ = next(iter(args.dataloader_train))
    if not args.resnet_cpu:
        ims = ims.to(args.device, non_blocking=True)
    visual_emb = resnet(ims)
    b, c, h, w = visual_emb.shape
    #visual_emb = visual_emb.view(b, c * h * w)
    visual_emb = visual_emb.view(b, c, h * w).transpose(1, 2)

    if args.resnet_cpu:
        visual_emb = visual_emb.to(args.device, non_blocking=True)
    #args.writer.add_graph(encoder, visual_emb)
    #args.writer.add_graph(decoder, encoder(visual_emb))

    # parameters
    params = list(decoder.parameters()) + list(encoder.parameters())
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)

    params_D = list(D.parameters())
    opt_D = torch.optim.Adam(params_D, lr=0.001, weight_decay=1e-3)

    # set precision
    if args.fp16:
        models_fp, optimizer = amp.initialize(models=[encoder, decoder],
                                              optimizers=optimizer,
                                              opt_level=args.fp16_opt_level)
        amp._amp_state.loss_scalers[0]._loss_scale = 2**20
        encoder, decoder = models_fp

    encoder.train()
    decoder.train()
    D.train()
    resnet.eval()

    # summary
    logger.debug("encoder: ")
    summary(encoder)
    logger.debug("decoder: ")
    summary(decoder)
    logger.debug("discriminator: ")
    summary(D)

    losses = AverageMeter()
    losses_G = AverageMeter()
    losses_D = AverageMeter()
    losses_gp = AverageMeter()
    loss_min = 1000
    step_global = 0

    value_real = 0.8
    value_fake = 0.2
    w_g = 1  # weight for GAN loss
    lambda_1 = 10
    criterion_GAN = nn.BCELoss()

    # weight for cross entropy
    ce_weight = torch.tensor(args.list_weight).to(args.device,
                                                  non_blocking=True)

    while (step_global < args.step_max):
        for (feature, y_in, lengths, indices) in args.dataloader_train:

            img_grid = torchvision.utils.make_grid(feature, nrow=8)
            args.writer.add_image('images_train',
                                  img_tensor=img_grid,
                                  global_step=step_global)

            with torch.no_grad():
                if not args.resnet_cpu:
                    feature = feature.to(args.device, non_blocking=True)
                visual_emb = resnet(feature)

            # skip last batch
            if visual_emb.shape[0] != batch_size:
                continue

            labels_real = torch.full((batch_size, ),
                                     value_real,
                                     device=args.device)
            labels_fake = torch.full((batch_size, ),
                                     value_fake,
                                     device=args.device)

            logger.debug("visual_emb {}".format(visual_emb.shape))
            b, c, h, w = visual_emb.shape
            # nchw to nte
            visual_token = visual_emb.view(b, c, h * w).transpose(1, 2)
            #visual_emb = visual_emb.view(b, args.dim_reformer * h * w)
            logger.debug("visual_emb {}".format(visual_token.shape))
            # 空間部分whがsequence次元に来て，channel部分がdim_enc次元に来たほうが良さそう

            if args.resnet_cpu:
                visual_emb = visual_emb.to(args.device, non_blocking=True)
                visual_token = visual_token.to(args.device, non_blocking=True)

            y_in = y_in.to(args.device, non_blocking=True)

            # run
            enc_keys = encoder(visual_token)
            logger.debug("enc_keys {}".format(enc_keys.shape))
            logger.debug(y_in.shape)
            y_out, loss_ce = decoder(y_in, return_loss=True, keys=enc_keys)

            logger.debug("y_in {}".format(y_in.shape))
            logger.debug("y_out {}".format(y_out.shape))

            # tag argmax
            b, l, c = y_out.shape
            weights = torch.softmax(y_out, dim=-1)  #
            indices_soft = (torch.arange(c).unsqueeze(0).unsqueeze(0).expand(
                weights.size())).to(args.device, non_blocking=True)
            y_out_indices = (weights * indices_soft).sum(dim=-1)

            #logger.info("y_out_indices.shape {}".format(y_out_indices.shape))

            # interpolated tag for gradient penalty
            # y_in[:, 1:]をone hotに変える
            # y_outとy_inをinterpolateし，y_hatとする
            # y_hatをラベル整数にする

            y_hat = F.one_hot(y_in[:, 1:].long(), num_classes=args.vocab_size)
            # ランダムな係数で補間する
            alpha_size = tuple((len(y_out), *(1, ) * (y_out.dim() - 1)))
            alpha_t = torch.Tensor
            alpha = alpha_t(*alpha_size).to(args.device).uniform_()
            y_hat = (y_hat.data * alpha + y_out.data * (1 - alpha))
            #logger.info("y_hat_indices {}".format(y_hat[0]))

            # make soft argmax
            b, l, c = y_hat.shape
            weights = torch.softmax(y_hat, dim=-1)  #
            indices_soft = (torch.arange(c).unsqueeze(0).unsqueeze(0).expand(
                weights.size())).to(args.device, non_blocking=True)
            y_hat_indices = (weights * indices_soft).sum(dim=-1)

            # logger.info("y_indices {}".format(y_in[0, 1:]))
            # logger.info("y_hat_indices {}".format(y_hat_indices[0]))
            # logger.info("y_out_indices {}".format(y_out_indices[0]))

            # loss G
            validity_fake = D(visual_emb, y_out_indices.long())
            logger.debug("validity_fake.view(-1) {}".format(
                validity_fake.view(-1).shape))
            loss_g = w_g * criterion_GAN(validity_fake.view(-1), labels_real)

            loss = loss_ce + loss_g
            logger.debug(loss.item())

            losses.update(loss_ce.item())
            losses_G.update(loss_g.item())

            # loss D
            validity_real = D(visual_emb, y_in[:, 1:])
            # Dのgradient penaltyを計算
            loss_gp = gradient_penalty(args, D, visual_emb, y_hat_indices,
                                       lambda_1)

            loss_D = criterion_GAN(
                validity_fake.view(-1), labels_fake) + criterion_GAN(
                    validity_real.view(-1), labels_real) + loss_gp

            losses_D.update(loss_D.item())
            losses_gp.update(loss_gp.item())

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
                loss_g = loss_g / args.gradient_accumulation_steps
                loss_D = loss_D / args.gradient_accumulation_steps

            # backward
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward(retain_graph=True)
                args.writer.add_scalar("train/loss",
                                       scalar_value=loss.item(),
                                       global_step=step_global)
                # G: only for visualization
                args.writer.add_scalar("train/loss_G",
                                       scalar_value=loss_g.item(),
                                       global_step=step_global)
                loss_D.backward()
                args.writer.add_scalar("train/loss_D",
                                       scalar_value=loss_D.item(),
                                       global_step=step_global)

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

                opt_D.step()
                opt_D.zero_grad()

            # logging
            if (step_global + 1) % args.step_log == 0:
                logger.info("steps: [#%d], loss_ce_train: %.4f" %
                            (step_global, losses.avg))
                logger.info("steps: [#%d], loss_G_train: %.4f" %
                            (step_global, losses_G.avg))
                logger.info("steps: [#%d], loss_D_train: %.4f" %
                            (step_global, losses_D.avg))
                logger.info("steps: [#%d], loss_gp_train: %.4f" %
                            (step_global, losses_gp.avg))

                losses.reset()
                losses_G.reset()
                losses_D.reset()

            # validation
            if (step_global + 1) % args.step_save == 0:

                loss_valid = validate(args.dataloader_valid, encoder, decoder,
                                      D, resnet, args, step_global, ce_weight)

                #if loss_valid < loss_min:
                if True:
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
                    torch.save(
                        D.state_dict(),
                        os.path.join(
                            args.model_path,
                            'discriminator_%d.pkl' % (step_global + 1)))

                #resnet.to(args.device)

            # end training
            if step_global == args.step_max:
                break
            else:
                step_global += 1

    args.writer.close()
    logger.info('done!')


def validate(dataloader,
             encoder,
             decoder,
             D,
             resnet,
             args,
             step,
             ce_weight=None):
    encoder.eval()
    decoder.eval()
    D.eval()

    eval_losses = AverageMeter()
    eval_losses_G = AverageMeter()
    eval_losses_D = AverageMeter()
    #eval_losses_gp = AverageMeter()

    value_real = 0.8
    value_fake = 0.2
    criterion_GAN = nn.BCELoss()

    for i, (feature, y_in, lengths, indices) in enumerate(dataloader):

        img_grid = torchvision.utils.make_grid(feature[:8], nrow=2)
        args.writer.add_image('images_val',
                              img_tensor=img_grid,
                              global_step=step + i)

        with torch.no_grad():
            if not args.resnet_cpu:
                feature = feature.to(args.device, non_blocking=True)
            visual_emb = resnet(feature)

        y_in = y_in.to(args.device, non_blocking=True)
        b, c, h, w = visual_emb.shape
        # nchw to nte
        visual_token = visual_emb.view(b, c, h * w).transpose(1, 2)
        #visual_emb = visual_emb.view(b, args.dim_reformer * h * w)

        labels_real = torch.full((batch_size, ),
                                 value_real,
                                 device=args.device)
        labels_fake = torch.full((batch_size, ),
                                 value_fake,
                                 device=args.device)

        if args.resnet_cpu:
            visual_emb = visual_emb.to(args.device, non_blocking=True)
            visual_token = visual_token.to(args.device, non_blocking=True)

        with torch.no_grad():
            # run
            enc_keys = encoder(visual_token)
            y_out, loss_ce = decoder(y_in, return_loss=True, keys=enc_keys)

            # tag argmax
            b, l, c = y_out.shape
            weights = torch.softmax(y_out, dim=-1)  #
            indices_soft = (torch.arange(c).unsqueeze(0).unsqueeze(0).expand(
                weights.size())).to(args.device, non_blocking=True)
            y_out_indices = (weights * indices_soft).sum(dim=-1)

            validity_fake = D(visual_emb, y_out_indices.long())
            loss_g = criterion_GAN(validity_fake.view(-1), labels_real[:b])

            #loss = loss_ce + loss_g
            eval_losses.update(loss_ce.item())
            eval_losses_G.update(loss_g.item())

            # for D
            validity_real = D(visual_emb, y_in[:, 1:])
            loss_D = criterion_GAN(validity_fake.view(-1),
                                   labels_fake[:b]) + criterion_GAN(
                                       validity_real.view(-1), labels_real[:b])
            eval_losses_D.update(loss_D.item())

            logger.debug("Loss: %.4f" % (eval_losses.avg))
            logger.debug("Loss_G: %.4f" % (eval_losses_G.avg))
            logger.debug("Loss_D: %.4f" % (eval_losses_D.avg))
            #logger.debug("Loss_gp: %.4f" % (eval_losses_gp.avg))

    logger.info("loss_ce_valid: %.4f" % (eval_losses.avg))
    logger.info("loss_G_valid: %.4f" % (eval_losses_G.avg))
    logger.info("loss_D_valid: %.4f" % (eval_losses_D.avg))
    #logger.info("Loss_gp: %.4f" % (eval_losses_gp.avg))
    args.writer.add_scalar("val/loss_ce",
                           scalar_value=eval_losses.avg,
                           global_step=step + i)
    args.writer.add_scalar("val/loss_G",
                           scalar_value=eval_losses_G.avg,
                           global_step=step + i)
    args.writer.add_scalar("val/loss_D",
                           scalar_value=eval_losses_D.avg,
                           global_step=step + i)
    # args.writer.add_scalar("val/loss_gp",
    #                        scalar_value=eval_losses_gp.avg,
    #                        global_step=step + i)

    encoder.train()
    decoder.train()
    D.train()

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
        #if step < 247: #batch_size=8
        #if step < 123: #batch_size=16
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

    # dataset
    batch_size = make_datasets(args)

    # model
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder, decoder, D, resnet = get_models(args)

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
        train(batch_size, encoder, decoder, D, resnet, args)
    else:
        test(encoder, decoder, resnet, args)
    elapsed_time = time.time() - start
    logger.info("elapsed_time:{0}".format(elapsed_time / 3600) + "[h]")
