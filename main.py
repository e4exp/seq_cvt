import os
import argparse
import time
from logging import getLogger, StreamHandler, DEBUG, INFO
logger = getLogger(__name__)

from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from nltk.translate.bleu_score import corpus_bleu
from torchsummary import summary
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
        depth=6,
        heads=8,
        max_seq_len=256  #4096
    )

    decoder = ReformerLM(num_tokens=args.vocab_size,
                         dim=args.dim_reformer,
                         depth=6,
                         heads=8,
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

    print(args.use_pretrain)
    if args.use_pretrain:
        trained_model_path = os.path.join(args.model_path,
                                          'decoder_pretrain_30000.pkl')
        decoder.load_state_dict(torch.load(trained_model_path))
        logger.info("loading model: {}".format(trained_model_path))
        print("loading model: {}".format(trained_model_path))

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
    dataset_train = ImageHTMLDataSet(args.data_dir_img, args.data_dir_html,
                                     args.data_path_csv_train, args.vocab,
                                     transform_train, resnet, "cpu",
                                     args.seq_len)
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
    dataset_valid = ImageHTMLDataSet(args.data_dir_img, args.data_dir_html,
                                     args.data_path_csv_valid, args.vocab,
                                     transform_valid, resnet, "cpu",
                                     args.seq_len)
    dataloader_valid = DataLoader(
        dataset=dataset_valid,
        batch_size=args.batch_size_val,
        shuffle=args.shuffle_test,
        num_workers=args.num_workers,
        collate_fn=dataset_valid.collate_fn_transformer)

    losses = AverageMeter()
    loss_min = 1000
    step_global = 0
    for epoch in range(args.step_max):
        for (visual_emb, y_in, lengths) in dataloader_train:
            visual_emb = visual_emb.to(args.device)
            # skip last batch
            if visual_emb.shape[0] != batch_size:
                continue
            y_in = y_in.to(args.device)

            b, s, c, h, w = visual_emb.shape
            # nchw to nte
            visual_emb = visual_emb.view(b, args.dim_reformer,
                                         c // args.dim_reformer * s * h *
                                         w).transpose(1, 2)
            #logger.debug("visual_emb {}".format(visual_emb.shape))

            # run
            enc_keys = encoder(visual_emb)
            #logger.debug(enc_keys.shape)
            #logger.debug(y_in.shape)
            loss = decoder(y_in, return_loss=True,
                           keys=enc_keys)  # (batch, seq, vocab)
            #logger.debug(y_out.shape)

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

            # end training
            step_global += 1

        # logging
        if (epoch + 1) % args.step_log == 0:
            logger.info("steps: [#%d], loss_train: %.4f" % (epoch, losses.avg))
            losses.reset()

        # validation
        if (epoch + 1) % args.step_save == 0:
            resnet.to("cpu")

            loss_valid = validate(dataloader_valid, encoder, decoder, args)
            if loss_valid < loss_min:
                loss_min = loss_valid
                # save models
                logger.info('=== saving models at epoch: {} ==='.format(epoch))
                torch.save(
                    decoder.state_dict(),
                    os.path.join(args.model_path,
                                 'decoder_%d.pkl' % (epoch + 1)))
                torch.save(
                    encoder.state_dict(),
                    os.path.join(args.model_path,
                                 'encoder_%d.pkl' % (epoch + 1)))

            #resnet.to(args.device)

    logger.info('done!')


def validate(dataloader, encoder, decoder, args):
    encoder.eval()
    decoder.eval()
    eval_losses = AverageMeter()

    for step, (visual_emb, y_in, lengths) in enumerate(dataloader):
        visual_emb = visual_emb.to(args.device)
        y_in = y_in.to(args.device)

        b, s, c, h, w = visual_emb.shape
        # nchw to nte
        visual_emb = visual_emb.view(b, args.dim_reformer,
                                     c // args.dim_reformer * s * h *
                                     w).transpose(1, 2)
        with torch.no_grad():
            # run
            enc_keys = encoder(visual_emb)
            loss = decoder(
                y_in,
                return_loss=True,
                keys=enc_keys,
            )  # (batch, seq, vocab)

            eval_losses.update(loss.item())
            logger.debug("Loss: %.4f" % (eval_losses.avg))
    logger.info("loss_valid: %.4f" % (eval_losses.avg))

    encoder.train()
    decoder.train()

    return eval_losses.avg


def predict(dataloader, encoder, decoder, args):
    encoder.eval()
    decoder.eval()

    tags_pred = []
    tags_gt = []

    pad = args.vocab('__PAD__')
    bgn = args.vocab('__BGN__')
    end = args.vocab('__END__')
    cnt = 0

    # when evaluating, just use the generate function, which will default to top_k sampling with temperature of 1.
    initial = torch.tensor([[bgn]]).long().repeat([args.batch_size_val,
                                                   1]).to(args.device)
    for step, (visual_emb, y_in, lengths) in enumerate(tqdm(dataloader)):
        #if step == 2:
        #    break
        visual_emb = visual_emb.to(args.device)
        y_in = y_in.to('cpu')
        b, s, c, h, w = visual_emb.shape
        # nchw to nte
        visual_emb = visual_emb.view(b, args.dim_reformer,
                                     c // args.dim_reformer * s * h *
                                     w).transpose(1, 2)

        with torch.no_grad():
            # generate text
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
                # preserve prediction
                tags = [args.vocab.idx2word[str(bgn)]] + [
                    args.vocab.idx2word[str(int(x))]
                    for x in sample if not x == pad
                ]
                tags_pred.append(tags)

                # save file
                str_pred = "\n".join(tags)
                path = os.path.join(args.out_dir_pred, str(cnt) + "_pred.html")
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
                path = os.path.join(args.out_dir_gt, str(cnt) + "_gt.html")
                with open(path, "w") as f:
                    f.write(str_gt)

                cnt += 1

    return tags_pred, tags_gt


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
    dataset = ImageHTMLDataSet(args.data_dir_img_test, args.data_dir_html_test,
                               args.data_path_csv_test, args.vocab, transform,
                               resnet, "cpu", args.seq_len)
    dataloader = DataLoader(dataset=dataset,
                            batch_size=args.batch_size_val,
                            shuffle=args.shuffle_test,
                            num_workers=args.num_workers,
                            collate_fn=dataset.collate_fn_transformer)

    tags_pred, tags_gt = predict(dataloader, encoder, decoder, args)

    # calc scores
    bleu = corpus_bleu(tags_gt, tags_pred)
    err = error_exact(tags_gt, tags_pred)
    acc = accuracy_exact(tags_gt, tags_pred)

    logger.info("bleu score: {}".format(bleu))
    logger.info("error : {}".format(err))
    logger.info("accuracy: {}".format(acc))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", default="031_reformer_ds")
    parser.add_argument("--data_name", default="014_flat_seq")
    parser.add_argument("--ckpt_name", default="ckpt")
    parser.add_argument("--mode", default="train")
    parser.add_argument("--step_load", type=int, default=0)
    parser.add_argument("--step_max", type=int, default=100)
    parser.add_argument("--step_log", type=int, default=1)
    parser.add_argument("--step_save", type=int, default=10)

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
