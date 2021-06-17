import os
import math
import json
from collections import OrderedDict
decoder = json.JSONDecoder(object_hook=None, object_pairs_hook=OrderedDict)
from logging import getLogger, StreamHandler, DEBUG, INFO
logger = getLogger(__name__)
from multiprocessing import Pool

from tqdm import tqdm
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from numpy.core.fromnumeric import mean, std
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

from models.vocab import build_vocab_from_list, build_vocab


def make_datasets(args, ):

    if args.mode == "train":
        p_aug_0 = 0.5
        p_aug = 0.25
        aug_albu = A.Compose(
            [
                A.GaussianBlur(p=p_aug),
                A.GaussNoise(p=p_aug),
                A.JpegCompression(p=p_aug),
                A.MultiplicativeNoise(elementwise=True, p=p_aug),
                #A.Downscale(p=p_aug),
                A.HueSaturationValue(p=p_aug),
                A.RGBShift(p=p_aug),
                A.ChannelShuffle(p=p_aug),
                A.ToGray(p=p_aug),
                A.ToSepia(p=p_aug),
                A.InvertImg(p=p_aug),
                A.RandomBrightnessContrast(p=p_aug),
                A.CLAHE(p=p_aug),
            ],
            p=p_aug_0)

        def transform_albu(image, transform=aug_albu):
            if transform:
                image_np = np.array(image)
                augmented = transform(image=image_np)
                image = Image.fromarray(augmented['image'])
            return image

        # train data
        # jitter_color = transforms.RandomApply([
        #     transforms.ColorJitter(
        #         brightness=(0, 1), contrast=(0, 1), saturation=(0, 1),
        #         hue=0.5),
        # ],
        #                                       p=0.5)
        transform_train = transforms.Compose([
            transforms.Lambda(transform_albu),
            #jitter_color,
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        dataset_train = ImageHTMLDataSet(args.data_dir_img,
                                         args.data_dir_html,
                                         args.data_dir_attr,
                                         args.data_path_csv_train,
                                         transform_train,
                                         args,
                                         flg_make_vocab=True)
        # validation data
        transform_valid = transforms.Compose([
            #transforms.RandomResizedCrop(args.crop_size,
            #                             scale=(1.0, 1.0),
            #                             ratio=(1.0, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        dataset_valid = ImageHTMLDataSet(args.data_dir_img, args.data_dir_html,
                                         args.data_dir_attr,
                                         args.data_path_csv_valid,
                                         transform_valid, args)

        batch_size = args.batch_size // args.gradient_accumulation_steps
        args.dataloader_train = DataLoader(dataset=dataset_train,
                                           batch_size=batch_size,
                                           shuffle=args.shuffle_train,
                                           num_workers=args.num_workers,
                                           pin_memory=True,
                                           collate_fn=collate_fn_transformer)
        args.dataloader_valid = DataLoader(dataset=dataset_valid,
                                           batch_size=args.batch_size_val,
                                           shuffle=args.shuffle_test,
                                           num_workers=args.num_workers,
                                           pin_memory=True,
                                           collate_fn=collate_fn_transformer)
    elif args.mode == "test":
        # vocab
        args.vocab = build_vocab(args.path_vocab_txt, args.path_vocab_w2i,
                                 args.path_vocab_i2w)
        batch_size = args.batch_size_val

        # dataset
        transform = transforms.Compose([
            #transforms.RandomResizedCrop(args.crop_size,
            #                             scale=(1.0, 1.0),
            #                             ratio=(1.0, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        dataset_test = ImageHTMLDataSet(args.data_dir_img_test,
                                        args.data_dir_html_test,
                                        args.data_dir_attr_test,
                                        args.data_path_csv_test, transform,
                                        args)
        args.dataloader_test = DataLoader(dataset=dataset_test,
                                          batch_size=args.batch_size_val,
                                          shuffle=args.shuffle_test,
                                          num_workers=args.num_workers,
                                          pin_memory=True,
                                          collate_fn=collate_fn_transformer)
    args.vocab_size = len(args.vocab)

    return batch_size


def wrapper_get_indices(args):
    return get_indices(*args)


def get_indices(id, tags, tags_open, tags_close, idx_open, idx_close,
                idx_other, idx_no_pair, len_tag_max):
    indices_is_close = []
    indices_idx_pair = []
    stack_open = []

    for i, tag in enumerate(tags):
        if tag in tags_open:
            # open tag
            stack_open.append([tag, i / len_tag_max])
            indices_is_close.append(idx_open)
            indices_idx_pair.append(idx_no_pair)
        elif tag in tags_close:
            # close tag
            indices_is_close.append(idx_close)
            if len(stack_open) != 0 and stack_open[-1][0] == tag.replace(
                    "/", ""):
                # matched case
                indices_idx_pair.append(stack_open[-1][1])
            else:
                # error case
                indices_idx_pair.append(idx_no_pair)
        else:
            # single tag
            indices_is_close.append(idx_other)
            indices_idx_pair.append(idx_no_pair)

    return id, indices_is_close, indices_idx_pair


class ImageHTMLDataSet(Dataset):
    def __init__(self,
                 data_dir_img,
                 data_dir_html,
                 data_dir_attr,
                 data_path_csv,
                 transform,
                 args,
                 flg_make_vocab=False):
        self.data_dir_img = data_dir_img
        self.data_dir_html = data_dir_html
        self.data_dir_attr = data_dir_attr
        self.transform = transform
        self.w_fix = 1500
        self.image_white = Image.new("RGB", (self.w_fix, self.w_fix),
                                     (255, 255, 255))
        self.max_num_divide_h = 8
        self.h_fix = self.w_fix * self.max_num_divide_h

        self.paths_image = []
        self.htmls = []
        self.attrs = []
        self.len_tag_max = args.seq_len
        self.str_undefined = "<undefined>"

        self.tag_bgn = "__BGN__"
        self.tag_end = "__END__"
        self.tag_unk = "__UNK__"
        self.tag_pad = "__PAD__"

        self.tags_single = [
            self.tag_bgn, self.tag_end, self.tag_unk, self.tag_pad, "text",
            "br", "img", "hr", "meta", "input", "embed", "area", "base", "col",
            "keygen", "link", "param", "source", "doctype"
        ]

        # fetch all paths
        with open(data_path_csv, "r") as f:
            lines = f.readlines()

        words = []
        stat_w = []
        stat_h = []
        stat_tags = []
        str_csv_new = ""
        # set file paths
        for line in lines:
            name_img, html = line.replace("\n", "").split(", ")
            #name_img = os.path.splitext(os.path.basename(path_img))[0]

            # check image size
            path_img = os.path.join(self.data_dir_img, name_img + ".png")
            img = Image.open(path_img)
            w, h = img.size
            #if h / w > self.max_num_divide_h:
            if h > self.h_fix:
                continue
            if w > self.w_fix:
                continue
            if not os.path.isfile(path_img):
                continue

            # append html tags
            html = html.split(" ")

            # attr
            # path_attr = os.path.join(self.data_dir_attr,
            #                          name_img + "_node_attr.txt")

            # with open(path_attr, "r") as f:
            #     attr = f.read().splitlines()

            # filter duplicated text
            html_new = []
            #attr_new = []
            tag_prev = ""
            for i, tag in enumerate(html):
                tag = tag.lower().strip()

                #at = attr[i]
                if tag == "text" and tag == tag_prev:
                    continue
                else:
                    html_new.append(tag)
                    #attr_new.append(at)
                tag_prev = tag
            html = html_new
            #attr = attr_new

            if len(html) > self.len_tag_max - 2:  # 2 considers BGN and END
                continue

            stat_w.append(w)
            stat_h.append(h)
            stat_tags.append(len(html))

            # append image filename
            self.paths_image.append(path_img)
            self.htmls.append(html)
            #self.attrs.append(attr)

            # new csv
            str_csv_new += name_img + ", "
            str_csv_new += " ".join(html) + "\n"

            # vocab
            words.extend(html)

        # write new csv
        name, ext = os.path.splitext(os.path.basename(data_path_csv))
        path_csv_new = data_path_csv.replace(name, name + "_new")
        with open(path_csv_new, "w") as f:
            f.write(str_csv_new)

        # make vocab
        if flg_make_vocab:
            args.vocab, args.list_weight = build_vocab_from_list(
                words, args, len(self.paths_image))

        # TODO: move this to vocab
        # filter out single tags
        words = [args.vocab.idx2word[str(i)] for i in range(len(args.vocab))]
        tags_target = list(
            filter(
                lambda x: True if x.replace("/", "").replace(">", "").replace(
                    "<", "") not in self.tags_single else False, words))
        # collect tags have "/"
        tags_close = list(
            filter(lambda x: True if "/" in x else False, tags_target))
        args.tags_close = list(set(tags_close))
        args.tags_open = list(set([tag.replace("/", "")
                                   for tag in tags_close]))

        self.vocab = args.vocab
        self.tags_close = args.tags_close
        self.tags_open = args.tags_open

        # tags
        self.tags = []
        print("htmls -> tags")
        for html in tqdm(self.htmls):
            tags = [self.vocab(token) for token in html]
            self.tags.append(tags)

        # open/close
        print("open/close")
        self.idx_no_pair = -1
        self.idx_close = 2
        self.idx_open = 1
        self.idx_other = 0
        self.indices_is_close = []
        self.indices_idx_pair = []

        print("start multiprocessing")
        htmls = self.htmls
        n_process = 4

        print("tags_open")
        print(len(self.tags_open))
        print("tags_close")
        print(len(self.tags_close))

        # id, tags, tags_open, tags_close, idx_open, idx_close, idx_other, idx_no_pair, len_tag_max
        args_wrapped = [[
            i, tags, self.tags_open, self.tags_close, self.idx_open,
            self.idx_close, self.idx_other, self.idx_no_pair, self.len_tag_max
        ] for i, tags in enumerate(htmls)]
        p = Pool(n_process)
        #ret = p.map(wrapper_get_indices, args_wrapped)
        rets = []
        with tqdm(total=len(htmls)) as t:
            for ret in p.imap_unordered(wrapper_get_indices, args_wrapped):
                t.update(1)
                rets.append(ret)

        #print(ret)
        ret = sorted(rets, key=lambda x: x[0])
        for i in range(len(ret)):
            self.indices_is_close.append(ret[i][1])
            self.indices_idx_pair.append(ret[i][2])
        print(len(self.indices_is_close))
        print(len(self.indices_idx_pair))
        print("end multiprocessing")

        print("============")
        print(data_path_csv)
        print("============")
        print('Created dataset of ' + str(len(self)) + ' items from ' +
              data_dir_img)

        # stat

        def show_stat(list_target, name):
            print("=== stat of {} ===".format(name))
            print("max: ", max(list_target))
            print("min: ", min(list_target))
            print("mean: ", mean(list_target))
            print("std: ", std(list_target))
            print()

        show_stat(stat_h, "img_h")
        show_stat(stat_w, "img_w")
        show_stat(stat_tags, "tag_length")

        # for debug
        with open("_paths_image_train.txt", "w") as f:
            f.write("\n".join(self.paths_image))

    def __len__(self):
        return len(self.paths_image)

    def __getitem__(self, idx):
        path_img = self.paths_image[idx]
        #tags = self.htmls[idx]
        tags = self.tags[idx]
        #attr = self.attrs[idx]
        indices_is_close = self.indices_is_close[idx]
        indices_idx_pair = self.indices_idx_pair[idx]

        # get image
        image = Image.open(path_img).convert('RGB')

        # paste on center
        x = (self.w_fix - image.size[0]) // 2
        y = 0
        image_pad = Image.new("RGB", (self.w_fix, self.h_fix), (255, 255, 255))
        image_pad.paste(image, (x, y))
        image_pad = image_pad.resize((256, 256 * self.max_num_divide_h))
        feature = self.transform(image_pad)

        # tags
        # Convert caption (string) to list of vocab ID's
        #tags = [self.vocab(token) for token in tags]
        tags.insert(0, self.vocab('__BGN__'))
        tags.append(self.vocab('__END__'))
        tags = torch.Tensor(tags)

        indices_is_close.insert(0, self.idx_other)
        indices_is_close.append(self.idx_other)
        indices_is_close = torch.Tensor(indices_is_close)

        indices_idx_pair.insert(0, self.idx_no_pair)
        indices_idx_pair.append(self.idx_no_pair)
        indices_idx_pair = torch.Tensor(indices_idx_pair)

        # file name
        idx = torch.Tensor(torch.ones(1) * idx)

        return feature, tags, indices_is_close, indices_idx_pair, idx


def collate_fn_transformer(data):
    max_seq = 2048

    # Sort datalist by caption length; descending order
    data.sort(key=lambda data_pair: len(data_pair[1]), reverse=True)
    features, tags_batch, is_close_batch, idx_pair_batch, indices = zip(*data)

    # Merge images (from tuple of 3D Tensor to 4D Tensor)
    features = torch.stack(features, 0)
    indices = torch.stack(indices, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor)
    lengths = [len(tags) for tags in tags_batch]  # List of caption lengths
    #targets_t = torch.zeros(len(tags_batch), max(lengths)).long()
    targets_tag = torch.zeros(len(tags_batch), max_seq).long()
    targets_is_close = torch.zeros(len(tags_batch), max_seq).long()
    targets_idx_pair = torch.zeros(len(tags_batch), max_seq).long() - 1

    # 単純に各batchが同じ長さになるよう0埋めしている
    for i, tags in enumerate(tags_batch):

        end = lengths[i]
        targets_tag[i, :end] = tags[:end]
        targets_is_close[i, :end] = is_close_batch[i][:end]
        targets_idx_pair[i, :end] = idx_pair_batch[i][:end]

    return features, targets_tag, targets_is_close, targets_idx_pair, lengths, indices
