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

from models.vocab import build_vocab_from_list, build_vocab


def make_datasets(args, ):

    if args.mode == "train":
        # train data
        transform_train = transforms.Compose([
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
        # vocab
        args.vocab = build_vocab(args.path_vocab_txt, args.path_vocab_w2i,
                                 args.path_vocab_i2w)
        batch_size = args.batch_size_val
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
        #self.resnet = resnet
        #self.device = device
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
            html = list(map(lambda x: x.lower().strip(), html))
            if len(html) > self.len_tag_max:
                continue

            # append image filename
            self.paths_image.append(path)
            self.htmls.append(html)

            words.extend(html)

        # make vocab
        if flg_make_vocab:
            args.vocab = build_vocab_from_list(words, args)
        self.vocab = args.vocab

        print('Created dataset of ' + str(len(self)) + ' items from ' +
              data_dir_img)

    def __len__(self):
        return len(self.paths_image)

    def __getitem__(self, idx):
        path_img = self.paths_image[idx]
        tags = self.htmls[idx]

        # get image
        image = Image.open(path_img).convert('RGB')

        # =====
        # divide image
        # =====
        # w, h = image.size
        # list_image_divided = []
        # for i in range(self.max_num_divide_h):
        #     list_image_divided.append(self.transform(self.image_white.copy()))

        # # max is 8 (min_w = 1500)
        # num_divide_h = math.ceil(h / w)
        # h_divided = int(h / num_divide_h)
        # for i in range(num_divide_h):
        #     h_start = i * h_divided
        #     h_end = h_start + h_divided
        #     if h_end > h:
        #         im_crop = image.crop((0, h_start, w, h))
        #         # paste
        #         im_crop = Image.new("RGB", (w, w),
        #                             (255, 255, 255)).paste(im_crop)
        #     else:
        #         im_crop = image.crop((0, h_start, w, h_end))

        #     im_crop = self.transform(im_crop)
        #     list_image_divided[i] = im_crop

        # # feed image to resnet
        # ims = torch.cat(list_image_divided).reshape(
        #     len(list_image_divided),
        #     *list_image_divided[0].shape).to(self.device)
        # with torch.no_grad():
        #     feature = self.resnet(ims)

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
