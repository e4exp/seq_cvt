import os
import math
import json
from collections import OrderedDict
decoder = json.JSONDecoder(object_hook=None, object_pairs_hook=OrderedDict)
from logging import getLogger, StreamHandler, DEBUG, INFO
logger = getLogger(__name__)

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from numpy.core.fromnumeric import mean, std

from models.vocab import build_vocab_from_list, build_vocab


def make_datasets(args, ):

    if args.mode == "train":
        # train data
        jitter_color = transforms.RandomApply([
            transforms.ColorJitter(
                brightness=(0, 1), contrast=(0, 1), saturation=(0, 1),
                hue=0.5),
        ],
                                              p=0.5)
        transform_train = transforms.Compose([
            jitter_color,
            #transforms.RandomResizedCrop(args.crop_size,
            #                             scale=(1.0, 1.0),
            #                             ratio=(1.0, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        dataset_train = ImageHTMLDataSet(args.data_dir_img,
                                         args.data_dir_html,
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


class ImageHTMLDataSet(Dataset):
    def __init__(self,
                 data_dir_img,
                 data_dir_html,
                 data_path_csv,
                 transform,
                 args,
                 flg_make_vocab=False):
        self.data_dir_img = data_dir_img
        self.data_dir_html = data_dir_html
        self.transform = transform
        self.w_fix = 1500
        self.image_white = Image.new("RGB", (self.w_fix, self.w_fix),
                                     (255, 255, 255))
        self.max_num_divide_h = 8
        self.h_fix = self.w_fix * self.max_num_divide_h

        self.paths_image = []
        self.htmls = []
        self.len_tag_max = args.seq_len

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

            # check image size
            path = os.path.join(self.data_dir_img, name_img + ".png")
            img = Image.open(path)
            w, h = img.size
            #if h / w > self.max_num_divide_h:
            if h > self.h_fix:
                continue
            if w > self.w_fix:
                continue
            if not os.path.isfile(path):
                continue

            # append html tags
            html = html.split(" ")

            # filter duplicated text
            html_new = []
            tag_prev = ""
            for tag in html:
                tag = tag.lower().strip()
                if tag == "text" and tag == tag_prev:
                    continue
                else:
                    html_new.append(tag)
                tag_prev = tag
            html = html_new

            if len(html) > self.len_tag_max - 2:  # 2 considers BGN and END
                continue

            stat_w.append(w)
            stat_h.append(h)
            stat_tags.append(len(html))

            # append image filename
            self.paths_image.append(path)
            self.htmls.append(html)

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
        self.vocab = args.vocab

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

    def __len__(self):
        return len(self.paths_image)

    def __getitem__(self, idx):
        path_img = self.paths_image[idx]
        tags = self.htmls[idx]

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
        tags = [self.vocab(token) for token in tags]
        tags.insert(0, self.vocab('__BGN__'))
        tags.append(self.vocab('__END__'))
        tags = torch.Tensor(tags)

        # file name
        idx = torch.Tensor(torch.ones(1) * idx)

        return feature, tags, idx


def collate_fn(data):
    # Sort datalist by caption length; descending order
    data.sort(key=lambda data_pair: len(data_pair[1]), reverse=True)
    features, tags_batch = zip(*data)

    # Merge images (from tuple of 3D Tensor to 4D Tensor)
    features = torch.stack(features, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor)
    lengths = [len(tags) for tags in tags_batch]  # List of caption lengths
    targets_t = torch.zeros(len(tags_batch), max(lengths)).long()

    # 単純に各batchが同じ長さになるよう0埋めしている
    for i, seq in enumerate(tags_batch):
        t = seq
        end = lengths[i]
        targets_t[i, :end] = t[:end]

    return features, targets_t, lengths


def collate_fn_transformer(data):
    max_seq = 2048

    # Sort datalist by caption length; descending order
    data.sort(key=lambda data_pair: len(data_pair[1]), reverse=True)
    features, tags_batch, indices = zip(*data)

    # Merge images (from tuple of 3D Tensor to 4D Tensor)
    features = torch.stack(features, 0)
    indices = torch.stack(indices, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor)
    lengths = [len(tags) for tags in tags_batch]  # List of caption lengths
    #targets_t = torch.zeros(len(tags_batch), max(lengths)).long()
    targets_t = torch.zeros(len(tags_batch), max_seq).long()

    # 単純に各batchが同じ長さになるよう0埋めしている
    for i, seq in enumerate(tags_batch):
        t = seq
        end = lengths[i]
        targets_t[i, :end] = t[:end]

    return features, targets_t, lengths, indices
