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

parser = argparse.ArgumentParser()
parser.add_argument("--experiment_name", default="018_transformer_flat_p2c")
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

experiment_name = "026_reformer"
#data_name = "015_flat_seq_pix2code"
data_name = "014_flat_seq"
ckpt_name = "ckpt"
log = experiment_name + ".log"
step_load = 0
g_steps = "1"
mode = "train"
#mode="test"
#mode="extract"
dbg = 1
opt_level = "O2"

args = parser.parse_args([
                          "--experiment_name", experiment_name, \
                          "--data_name", data_name, \
                          "--ckpt_name", ckpt_name, \
                          "--mode", mode, \
                          "--step_load", step_load, \
                          #"--fp16", \
                          "--fp16_opt_level", opt_level, \
                          "--gradient_accumulation_steps", g_steps, \
                          ])

# Paths
root = "../drnn/experiments"
exp_root = root + "/" + args.experiment_name
data_root = root + "/" + args.data_name

data_dir = data_root + "/dataset"
data_dir_csv = exp_root + "/dataset"
# train
data_dir_img = data_dir + "/train/img"
data_dir_html = data_dir + "/train/html"
data_dir_feature = data_dir + "/train/feature"
# test
data_dir_img_test = data_dir + "/test/img"
data_dir_html_test = data_dir + "/test/html"
data_dir_feature_test = data_dir + "/test/feature"

# checkpoint
model_path = exp_root + "/" + args.ckpt_name

if not os.path.exists(exp_root):
    os.mkdir(exp_root)
if not os.path.exists(model_path):
    os.mkdir(model_path)

# Hyperparams
batch_size = 4
num_workers = 4
embed_size_tag = 32
num_epochs = args.num_epochs
learning_rate = 0.001
hidden_size = 512
num_layers = 1
seq_len = 4096

# Other params
shuffle_train = True
shuffle_test = False
use_feature = False  # use .png.npy if True
max_sample = seq_len  # for predictions

# Logging Variables
step_load = args.step_load
step_save = 20
step_log = 1

#crop_size = 224
crop_size = 256


def train():

    batch_size_ = batch_size // args.gradient_accumulation_steps

    # Fieldオブジェクトの作成
    def tokenize(text):
        return text.split(" ")

    BOS_WORD = '<s>'
    EOS_WORD = '</s>'
    BLANK_WORD = "<blank>"
    SRC = data.Field(sequential=False, use_vocab=False)
    TRG = data.Field(tokenize=tokenize,
                     init_token=BOS_WORD,
                     eos_token=EOS_WORD,
                     pad_token=BLANK_WORD,
                     fix_length=seq_len)

    # CSVファイルを読み込み、TabularDatasetオブジェクトの作成
    train, _ = data.TabularDataset.splits(
        path=data_dir_csv,
        #path=data_dir,
        train='train.csv',
        test='test.csv',
        format='csv',
        fields=[('src', SRC), ('trg', TRG)],
        filter_pred=lambda x: len(vars(x)['trg']) <= seq_len)

    # 可視化
    #for i, example in enumerate([(x.src, x.trg) for x in train[0:5]]):
    #    print(f"Example_{i}:{example}")

    MIN_FREQ = 1
    TRG.build_vocab(train.trg, min_freq=MIN_FREQ)
    train_iter = data.BucketIterator(train,
                                     batch_size=batch_size,
                                     repeat=False,
                                     sort_key=lambda x: len(x.trg))
    #val_iter = data.BucketIterator(val, batch_size=1, repeat=False, sort_key=lambda x: len(x.src))

    # 可視化
    if False:
        batch = next(iter(train_iter))
        src_matrix = batch.src.T
        print(src_matrix, src_matrix.size())
        print("---")
        trg_matrix = batch.trg.T
        print(trg_matrix, trg_matrix.size())

    vocab_size = len(TRG.vocab)
    logger.debug("vocab_size: {}".format(vocab_size))

    # define models
    #resnet = models.resnet50(pretrained=True)
    resnet = models.resnet18(pretrained=True)
    #resnet = Sequential(*list(resnet.children())[:-4]) # ([b, 512, 28, 28])
    #resnet = Sequential(*list(resnet.children())[:-2], nn.AdaptiveAvgPool2d((2, 2)))
    resnet = Sequential(*list(resnet.children())[:-2],
                        nn.AdaptiveAvgPool2d((4, 4)))

    #extractor = models.squeezenet1_0(pretrained=True)
    #extractor.classifier = nn.AdaptiveAvgPool2d((4, 4))

    dim_reformer = 256
    encoder = Reformer(
        dim=dim_reformer,
        depth=1,
        heads=1,
        max_seq_len=256  #4096
    )

    decoder = ReformerLM(num_tokens=vocab_size,
                         dim=dim_reformer,
                         depth=1,
                         heads=1,
                         max_seq_len=seq_len,
                         causal=True)

    # load models
    if step_load != 0:
        trained_model_path = os.path.join(model_path,
                                          'encoder_{}.pkl'.format(step_load))
        encoder.load_state_dict(torch.load(trained_model_path))
        logger.debug("loading model: {}".format(trained_model_path))

        trained_model_path = os.path.join(model_path,
                                          'decoder_{}.pkl'.format(step_load))
        decoder.load_state_dict(torch.load(trained_model_path))
        logger.debug("loading model: {}".format(trained_model_path))

    # loss
    criterion_ce = nn.CrossEntropyLoss()

    # parameters
    params = list(decoder.parameters()) + list(encoder.parameters())
    optimizer = torch.optim.Adam(params, lr=learning_rate)

    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder.to(device)
    decoder.to(device)
    #resnet.to(device)

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

    # feature
    # transform = transforms.ToTensor()
    transform = transforms.Compose([
        transforms.RandomResizedCrop(crop_size,
                                     scale=(1.0, 1.0),
                                     ratio=(1.0, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image_white = Image.new("RGB", (1500, 1500), (255, 255, 255))
    max_num_divide_h = 16

    #torch.autograd.set_detect_anomaly(True)
    for epoch in range(step_load, num_epochs + step_load):
        losses_t = []
        for step, batch in enumerate(train_iter):

            src = batch.src
            # skip last batch
            if src.shape[0] != batch_size:
                continue
            y_in = batch.trg.transpose(1, 0).to(device)

            torch.cuda.empty_cache()
            features = []
            for name in src:
                path = data_dir_img + "/" + str(name.item()) + ".png"
                logger.debug("img path: {}".format(path))
                image = Image.open(path).convert('RGB')

                # =====
                # divide image
                # =====
                w, h = image.size
                list_image_divided = []
                for i in range(max_num_divide_h):
                    list_image_divided.append(transform(image_white.copy()))

                # max is 18 (as min_w = 1500, max_h = 26000)
                num_divide_h = math.ceil(h / w)
                h_divided = int(h / num_divide_h)
                for i in range(num_divide_h):
                    h_start = i * h_divided
                    h_end = h_start + h_divided
                    if h_end > h:
                        im_crop = image.crop((0, h_start, w, h))
                        # paste
                        im_crop = Image.new("RGB", (w, w),
                                            (255, 255, 255)).paste(im_crop)
                    else:
                        im_crop = image.crop((0, h_start, w, h_end))

                    im_crop = transform(im_crop)
                    list_image_divided[i] = im_crop

                ims = torch.cat(list_image_divided).reshape(
                    len(list_image_divided),
                    *list_image_divided[0].shape).detach()  #.to(device)

                with torch.no_grad():
                    feature = resnet(ims)
                features.append(feature)

            visual_emb = torch.cat(features).reshape(
                len(features), *features[0].shape).to(device)
            logger.debug("visual_emb {}".format(visual_emb.shape))
            b, s, c, h, w = visual_emb.shape
            # nchw to nte
            #visual_emb = visual_emb.view(b, c, s * h * w).transpose(1, 2) # nchw to nte
            visual_emb = visual_emb.view(b, dim_reformer, c // dim_reformer *
                                         s * h * w).transpose(1, 2)
            logger.debug("visual_emb {}".format(visual_emb.shape))

            # run
            enc_keys = encoder(visual_emb)
            logger.debug(enc_keys.shape)
            logger.debug(y_in.shape)
            y_out = decoder(y_in, keys=enc_keys)  # (batch, seq, vocab)
            logger.debug(y_out.shape)

            loss = criterion_ce(
                y_out.reshape(batch_size * seq_len, vocab_size),
                y_in.reshape(batch_size * seq_len).long())
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
                # Zero out buffers
                #resnet.zero_grad()
                #encoder.zero_grad()
                #decoder.zero_grad()
                optimizer.zero_grad()

                logger.debug("zero_grad")

        if epoch % step_log == 0:
            logger.info("Epoch [#%d], Loss_t: %.4f" %
                        (epoch, np.mean(losses_t)))

        if (epoch + 1) % step_save == 0:
            # Save our models
            logger.info('!!! saving models at epoch: ' + str(epoch))
            torch.save(
                decoder.state_dict(),
                os.path.join(model_path, 'decoder_%d.pkl' % (epoch + 1)))
            torch.save(
                encoder.state_dict(),
                os.path.join(model_path, 'encoder_%d.pkl' % (epoch + 1)))

    logger.info('done!')


start = time.time()
train()

elapsed_time = time.time() - start
logger.info("elapsed_time:{0}".format(elapsed_time / 3600) + "[h]")
