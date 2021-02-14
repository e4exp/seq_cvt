import os
import math
import json
from collections import OrderedDict
decoder = json.JSONDecoder(object_hook=None, object_pairs_hook=OrderedDict)
from logging import getLogger, StreamHandler, DEBUG, INFO
logger = getLogger(__name__)

import torch
from PIL import Image
from torch.utils.data import Dataset


class ImageHTMLDataSet(Dataset):
    def __init__(self,
                 args,
                 data_path_csv,
                 data_dir_attr,
                 transform,
                 resnet,
                 device,
                 max_w_img=1500):
        self.data_dir_attr = data_dir_attr
        self.vocab = args.vocab
        self.transform = transform
        self.resnet = resnet
        self.device = device
        self.image_white = Image.new("RGB", (max_w_img, max_w_img),
                                     (255, 255, 255))
        self.max_num_divide_h = 16
        self.w_max_img = max_w_img
        self.h_max_img = self.w_max_img * self.max_num_divide_h
        self.max_len = args.seq_len
        self.str_undefined = "<undefined>"

        self.paths_image = []
        self.htmls = []
        self.attrs = []
        self.names = []

        # fetch all paths
        with open(data_path_csv, "r") as f:
            lines = f.readlines()

        # set file paths
        for line in lines:
            path_img, html = line.replace("\n", "").split(", ")
            name_img = os.path.splitext(os.path.basename(path_img))[0]

            # check image size
            img = Image.open(path_img)
            w, h = img.size
            if h / w > self.max_num_divide_h:
                continue

            # check html length
            html = html.split(" ")
            html = list(map(lambda x: x.lower().strip(), html))
            if len(html) > self.max_len:
                continue

            # append data
            self.paths_image.append(path_img)
            self.htmls.append(html)
            path_attr = os.path.join(self.data_dir_attr,
                                     name_img + "_node_attr.txt")
            with open(path_attr, "r") as f:
                attr = f.readlines()
            self.attrs.append(attr)
            self.names.append(name_img)

        print('Created dataset of ' + str(len(self)) + "items")

    def __len__(self):
        return len(self.paths_image)

    def __getitem__(self, idx):
        path_img = self.paths_image[idx]
        tags = self.htmls[idx]
        attr = self.attrs[idx]

        # get image
        image = Image.open(path_img).convert('RGB')

        # =====
        # divide image
        # =====
        w, h = image.size
        list_image_divided = []
        for i in range(self.max_num_divide_h):
            list_image_divided.append(self.transform(self.image_white.copy()))

        # max is 16 (min_w = 1500, max_h = 24000)
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

            im_crop = self.transform(im_crop)
            list_image_divided[i] = im_crop

        # feed image to resnet
        ims = torch.cat(list_image_divided).reshape(
            len(list_image_divided),
            *list_image_divided[0].shape).to(self.device)
        with torch.no_grad():
            feature = self.resnet(ims)

        # tags
        # Convert caption (string) to list of vocab ID's
        tags = [self.vocab(token) for token in tags]
        tags.insert(0, self.vocab('__BGN__'))
        tags.append(self.vocab('__END__'))
        tags = torch.Tensor(tags)

        # attr
        # return normalized relative position of the tag
        attr_normed = []
        for i, at in enumerate(attr):
            if i == 0 or i == len(attr) - 1:
                # html
                attr_cx = self.w_max_img // 2
                attr_cy = self.h_max_img // 2
                attr_w = 1
                attr_h = 1
            else:
                at = json.loads(at)
                position = at["position"]
                visibility = at["visibility"]

                if int(visibility) == 1:
                    # visible
                    top = int(position["top"])
                    bottom = int(position["bottom"])
                    left = int(position["left"])
                    right = int(position["right"])
                    if top == self.str_undefined or bottom == self.str_undefined or left == self.str_undefined or right == self.str_undefined:
                        attr_cx = 0
                        attr_cy = 0
                        attr_w = 0
                        attr_h = 0
                    else:
                        attr_cx = ((left + right) // 2) / self.w_max_img
                        attr_cy = ((top + bottom) // 2) / self.h_max_img
                        attr_w = (right - left) / self.w_max_img
                        attr_h = (bottom - top) / self.h_max_img
                else:
                    # invisible
                    attr_cx = 0
                    attr_cy = 0
                    attr_w = 0
                    attr_h = 0

            at_normed = [attr_cx, attr_cy, attr_w, attr_h]
            attr_normed.append(at_normed)
        # bgn/end
        attr_normed.insert(0, [0, 0, 0, 0])
        attr_normed.append([0, 0, 0, 0])
        attr = torch.Tensor(attr_normed)

        return idx, feature, tags, attr,

    def collate_fn_transformer(self, data):
        max_seq = self.max_len

        # Sort datalist by caption length; descending order
        data.sort(key=lambda data_pair: len(data_pair[2]), reverse=True)
        indices, features, tags_batch, attrs_batch = zip(*data)

        # Merge images (from tuple of 3D Tensor to 4D Tensor)
        features = torch.stack(features, 0)

        # Merge captions (from tuple of 1D tensor to 2D tensor)
        lengths = [len(tags) for tags in tags_batch]  # List of caption lengths
        #targets_t = torch.zeros(len(tags_batch), max(lengths)).long()
        targets_tag = torch.zeros(len(tags_batch), max_seq).long()
        targets_attr = torch.Tensor([0, 0, 0,
                                     0]).repeat(len(tags_batch), max_seq, 1)

        # 単純に各batchが同じ長さになるよう0埋めしている
        for i, tags in enumerate(tags_batch):
            end = lengths[i]
            targets_tag[i, :end] = tags[:end]
            targets_attr[i, :end] = attrs_batch[i][:end]

        return indices, features, targets_tag, targets_attr
