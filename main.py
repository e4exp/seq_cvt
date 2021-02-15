import os
import argparse
import time
import numpy as np
from logging import getLogger, StreamHandler, DEBUG, INFO
logger = getLogger(__name__)

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from nltk.translate.bleu_score import corpus_bleu
from torchsummary import summary
from torchvision import models
from torch.nn import Sequential
from reformer_pytorch import Reformer, ReformerLM
from reformer_pytorch.generative_tools import top_k
#from apex import amp
#from apex.parallel import DistributedDataParallel as DDP
from reformer_pytorch.generative_tools import TrainingWrapper
from torch.multiprocessing import set_start_method
try:
    set_start_method('spawn')
except RuntimeError:
    pass

from models.dataset import ImageHTMLDataSet
from models.vocab import build_vocab
from models.metrics import error_exact, accuracy_exact


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
    resnet = Sequential(*list(resnet.children())[:-2],
                        nn.AdaptiveAvgPool2d((4, 4)))

    encoder = Reformer(
        dim=args.dim_reformer,
        depth=1,
        heads=1,
        max_seq_len=256  #4096
    )

    decoder = ReformerLM(num_tokens=args.vocab_size + args.attribute_size,
                         dim=args.dim_reformer,
                         depth=1,
                         heads=1,
                         max_seq_len=args.seq_len,
                         causal=True)
    pad = args.vocab('__PAD__')
    #decoder = TrainingWrapper(decoder, ignore_index=pad, pad_value=pad)
    decoder = TrainingWrapper(decoder, pad_value=pad)

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

    if args.use_pretrain:
        trained_model_path = os.path.join(args.model_path,
                                          'decoder_pretrain_30000.pkl')
        decoder.load_state_dict(torch.load(trained_model_path))
        logger.info("loading model: {}".format(trained_model_path))

    # set device
    encoder.to(args.device)
    decoder.to(args.device)
    #resnet.to(args.device)

    return encoder, decoder, resnet


def train(encoder, decoder, resnet, args):

    batch_size = args.batch_size // args.gradient_accumulation_steps

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

    # train data
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(args.crop_size,
                                     scale=(1.0, 1.0),
                                     ratio=(1.0, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    dataset_train = ImageHTMLDataSet(
        args,
        args.data_path_csv_train,
        args.data_dir_attr,
        transform_train,
        resnet,
        "cpu",
    )
    dataloader_train = DataLoader(
        dataset=dataset_train,
        batch_size=batch_size,
        shuffle=args.shuffle_train,
        num_workers=args.num_workers,
        collate_fn=dataset_train.collate_fn_transformer)
    # validation data
    transform_valid = transforms.Compose([
        transforms.RandomResizedCrop(args.crop_size,
                                     scale=(1.0, 1.0),
                                     ratio=(1.0, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    dataset_valid = ImageHTMLDataSet(args, args.data_path_csv_valid,
                                     args.data_dir_attr, transform_valid,
                                     resnet, "cpu")
    dataloader_valid = DataLoader(
        dataset=dataset_valid,
        batch_size=args.batch_size_val,
        shuffle=args.shuffle_test,
        num_workers=args.num_workers,
        collate_fn=dataset_valid.collate_fn_transformer)

    losses_tag = AverageMeter()
    losses_attr = AverageMeter()
    loss_min = 1000
    step_global = 0
    while (step_global < args.step_max):
        for (idx, visual_emb, y_tag, y_attr) in dataloader_train:
            visual_emb = visual_emb.to(args.device)
            # skip last batch
            if visual_emb.shape[0] != batch_size:
                continue
            y_tag = y_tag.to(args.device)
            y_attr = y_attr.to(args.device)

            b, s, c, h, w = visual_emb.shape
            # nchw to nte
            visual_emb = visual_emb.view(b, args.dim_reformer,
                                         c // args.dim_reformer * s * h *
                                         w).transpose(1, 2)
            # run
            enc_keys = encoder(visual_emb)

            xi_tag = y_tag[:, :-1]
            xo_tag = y_tag[:, 1:]
            logger.debug("y_attr: {}".format(y_attr.shape))

            # (batch, seq, vocab)
            out = decoder(xi_tag, keys=enc_keys)
            logger.debug("out: {}".format(out.shape))
            # swap axis to (batch, vocab, seq)
            out = out.transpose(1, 2)
            out_tag = out[:, :args.vocab_size, :]
            out_attr = out[:, args.vocab_size:, :].transpose(2, 1)

            logger.debug("out: {}".format(out.shape))
            logger.debug("out_attr: {}".format(out_attr.shape))
            loss_tag = F.cross_entropy(out_tag, xo_tag)
            loss_attr = F.l1_loss(out_attr, y_attr[:, 1:, :])
            loss = loss_tag + loss_attr

            #logger.debug(loss.item())
            losses_tag.update(loss_tag.item())
            losses_attr.update(loss_attr.item())

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
                logger.info("steps: [#%d], loss_tag: %.4f loss_attr: %.4f" %
                            (step_global, losses_tag.avg, losses_attr.avg))
                losses_tag.reset()
                losses_attr.reset()

            # validation
            if (step_global + 1) % args.step_save == 0:
                resnet.to("cpu")

                loss_valid = validate(dataloader_valid, encoder, decoder, args)
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


def validate(dataloader, encoder, decoder, args):
    encoder.eval()
    decoder.eval()
    eval_losses = AverageMeter()

    with torch.no_grad():
        for step, (idx, visual_emb, y_tag, y_attr) in enumerate(dataloader):
            visual_emb = visual_emb.to(args.device)
            y_tag = y_tag.to(args.device)
            y_attr = y_attr.to(args.device)

            b, s, c, h, w = visual_emb.shape
            # nchw to nte
            visual_emb = visual_emb.view(b, args.dim_reformer,
                                         c // args.dim_reformer * s * h *
                                         w).transpose(1, 2)

            # run
            enc_keys = encoder(visual_emb)
            xi_tag = y_tag[:, :-1]
            xo_tag = y_tag[:, 1:]
            out = decoder(xi_tag, keys=enc_keys)
            out = out.transpose(1, 2)
            out_tag = out[:, :args.vocab_size, :]
            out_attr = out[:, args.vocab_size:, :].transpose(2, 1)

            loss_tag = F.cross_entropy(out_tag, xo_tag)
            loss_attr = F.l1_loss(out_attr, y_attr[:, 1:, :])
            loss = loss_tag + loss_attr

            eval_losses.update(loss.item())
            logger.debug("Loss: %.4f" % (eval_losses.avg))

    logger.info("loss_valid: %.4f" % (eval_losses.avg))

    encoder.train()
    decoder.train()

    return eval_losses.avg


@torch.no_grad()
def generate(args,
             net,
             start_tokens,
             seq_len,
             eos_token=None,
             temperature=1.,
             filter_logits_fn=top_k,
             filter_thres=0.9,
             **kwargs):

    num_dims = len(start_tokens.shape)
    if num_dims == 1:
        start_tokens = start_tokens[None, :]

    b, t = start_tokens.shape

    net.eval()
    out = start_tokens
    input_mask = kwargs.pop('input_mask', None)

    if input_mask is None:
        input_mask = torch.full_like(out,
                                     True,
                                     dtype=torch.bool,
                                     device=out.device)
    out_attr = []
    for _ in range(seq_len):
        x = out[:, -args.seq_len:]
        input_mask = input_mask[:, -args.seq_len:]

        logits = net(x, input_mask=input_mask, **kwargs)[:, -1, :]
        # get vocab dim

        #print("logits: {}".format(logits.shape))
        logits_tag = logits[:, :args.vocab_size]
        logits_attr = logits[:, args.vocab_size:].to('cpu')
        out_attr.append(logits_attr)

        filtered_logits = filter_logits_fn(logits_tag, thres=filter_thres)
        probs = F.softmax(filtered_logits / temperature, dim=-1)
        sample = torch.multinomial(probs, 1)

        out = torch.cat((out, sample), dim=-1)
        input_mask = F.pad(input_mask, (0, 1), value=True)

        if eos_token is not None and (sample == eos_token).all():
            break

    out = out[:, t:]

    if num_dims == 1:
        out = out.squeeze(0)

    out_attr = torch.cat(out_attr).reshape(len(out_attr), *out_attr[0].shape)
    print(out_attr.shape)

    return out, out_attr


def predict(dataset, dataloader, encoder, decoder, args):
    encoder.eval()
    decoder.eval()

    tags_pred = []
    tags_gt = []
    losses_attr = []

    pad = args.vocab('__PAD__')
    bgn = args.vocab('__BGN__')
    end = args.vocab('__END__')

    for step, (indices, visual_emb, y_tag,
               y_attr) in enumerate(tqdm(dataloader)):
        #if step == 1:
        #    break
        #for idx in indices:
        #    name = dataset.names[idx]
        #    print("name: ", name)

        visual_emb = visual_emb.to(args.device)
        y_tag = y_tag.to('cpu')
        y_attr = y_attr.to('cpu')

        b, s, c, h, w = visual_emb.shape
        # nchw to nte
        visual_emb = visual_emb.view(b, args.dim_reformer,
                                     c // args.dim_reformer * s * h *
                                     w).transpose(1, 2)
        # when evaluating, just use the generate function, which will default to top_k sampling with temperature of 1.
        initial = torch.tensor([[bgn]]).long().repeat([b, 1]).to(args.device)

        with torch.no_grad():
            # generate text
            enc_keys = encoder(visual_emb)
            samples, attrs = generate(
                args,
                decoder,
                initial,
                args.seq_len,
                eos_token=end,
                keys=enc_keys,
            )  # assume end token is 1, or omit and it will sample up to 100
            #logger.debug("generated sentence: {}".format(samples))
            #logger.debug("ground truth: {}".format(y_tag))
            logger.debug("generated sentence: {}".format(samples.shape))
            logger.debug("generated attr: {}".format(attrs.shape))
            logger.debug("ground truth: {}".format(y_tag.shape))
            logger.debug("ground truth attr: {}".format(y_attr.shape))

            attrs = attrs.transpose(0, 1)
            loss_attr = F.l1_loss(attrs, y_attr).detach().numpy().copy()
            losses_attr.append(loss_attr)

            samples = samples.to('cpu').detach().numpy().copy()
            str_pred = ""
            str_gt = ""

            # iterate over batch
            for i, sample in enumerate(samples):
                name = dataset.names[indices[i]]

                # preserve prediction
                tags = [args.vocab.idx2word[str(bgn)]] + [
                    args.vocab.idx2word[str(int(x))]
                    for x in sample if not x == pad
                ]
                tags_pred.append(tags)

                # save file
                str_pred = "\n".join(tags)
                path = os.path.join(args.out_dir_pred, name + "_pred.html")
                with open(path, "w") as f:
                    f.write(str_pred)

                # preserve ground truth
                gt = [
                    args.vocab.idx2word[str(int(x))] for x in y_tag[i]
                    if not x == pad
                ]
                tags_gt.append([gt])

                # save file
                str_gt = "\n".join(gt)
                path = os.path.join(args.out_dir_gt, name + "_gt.html")
                with open(path, "w") as f:
                    f.write(str_gt)

    return tags_pred, tags_gt, losses_attr


def test(encoder, decoder, resnet, args):

    # dataset
    transform = transforms.Compose([
        transforms.RandomResizedCrop(args.crop_size,
                                     scale=(1.0, 1.0),
                                     ratio=(1.0, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    dataset = ImageHTMLDataSet(args, args.data_path_csv_test,
                               args.data_dir_attr_test, transform, resnet,
                               "cpu")
    dataloader = DataLoader(dataset=dataset,
                            batch_size=args.batch_size_val,
                            shuffle=args.shuffle_test,
                            num_workers=args.num_workers,
                            collate_fn=dataset.collate_fn_transformer)

    tags_pred, tags_gt, losses_attr = predict(dataset, dataloader, encoder,
                                              decoder, args)

    # calc scores
    bleu = corpus_bleu(tags_gt, tags_pred)
    err = error_exact(tags_gt, tags_pred)
    acc = accuracy_exact(tags_gt, tags_pred)
    l1 = np.mean(losses_attr)

    logger.info("bleu score: {}".format(bleu))
    logger.info("error : {}".format(err))
    logger.info("accuracy: {}".format(acc))
    logger.info("l1 loss: {}".format(l1))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", default="031_reformer_ds")
    parser.add_argument("--data_name", default="014_flat_seq")
    parser.add_argument("--ckpt_name", default="ckpt")
    parser.add_argument("--mode", default="train")
    parser.add_argument("--step_load", type=int, default=0)
    parser.add_argument("--step_max", type=int, default=10000)
    parser.add_argument("--step_log", type=int, default=100)
    parser.add_argument("--step_save", type=int, default=1000)

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
        default=8,
        help=
        "Number of updates steps to accumulate before performing a backward/update pass."
    )
    parser.add_argument("--max_grad_norm",
                        default=1.0,
                        type=float,
                        help="Max gradient norm.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--batch_size_val", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--log_level", type=str, default="DEBUG")
    parser.add_argument('--use_pretrain', action='store_true')

    args = parser.parse_args()

    # Paths
    root = "../drnn/experiments"
    exp_root = root + "/" + args.experiment_name
    data_root = root + "/" + args.data_name
    data_dir = data_root + "/dataset"

    # train
    args.data_dir_attr = data_dir + "/train/attr"
    # test
    args.data_dir_attr_test = data_dir + "/test/attr"
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
    args.dim_reformer = 256

    # Other params
    args.shuffle_train = True
    args.shuffle_test = False
    args.max_sample = args.seq_len  # for predictions
    args.crop_size = 256
    args.attribute_size = 4

    # vocabulary
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

    # log level
    log_level = args.log_level
    handler = StreamHandler()
    handler.setLevel(log_level)
    logger.setLevel(log_level)
    logger.addHandler(handler)
    logger.propagate = False

    start = time.time()
    logger.info("mode: {}".format(args.mode))
    # start training
    if args.mode == "train":
        train(encoder, decoder, resnet, args)
    else:
        test(encoder, decoder, resnet, args)
    elapsed_time = time.time() - start
    logger.info("elapsed_time:{0}".format(elapsed_time / 3600) + "[h]")
