import os
import math
import json
from collections import OrderedDict
decoder = json.JSONDecoder(object_hook=None, object_pairs_hook=OrderedDict)

import torch
from PIL import Image
from torch.utils.data import Dataset


class ImageHTMLDataSet(Dataset):
    def __init__(self, data_dir_img, data_dir_html, data_path_csv, vocab,
                 transform, resnet, device, max_len):
        self.data_dir_img = data_dir_img
        self.data_dir_html = data_dir_html
        self.vocab = vocab
        self.transform = transform
        self.resnet = resnet
        self.device = device
        self.image_white = Image.new("RGB", (1500, 1500), (255, 255, 255))
        self.max_num_divide_h = 16
        self.max_len = max_len

        self.paths_image = []
        self.htmls = []

        # fetch all paths
        with open(data_path_csv, "r") as f:
            lines = f.readlines()

        # set file paths
        for line in lines:
            name_img, html = line.replace("\n", "").split(", ")

            # check image size
            path = os.path.join(self.data_dir_img, name_img + ".png")
            img = Image.open(path)
            w, h = img.size
            if h / w > self.max_num_divide_h:
                continue

            # append html tags
            html = html.split(" ")
            html = list(map(lambda x: x.lower().strip(), html))
            if len(html) > self.max_len:
                continue

            # append data
            self.paths_image.append(path)
            self.htmls.append(html)

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

        return feature, tags

    def collate_fn_transformer(self, data):
        max_seq = self.max_len

        # Sort datalist by caption length; descending order
        data.sort(key=lambda data_pair: len(data_pair[1]), reverse=True)
        features, tags_batch = zip(*data)

        # Merge images (from tuple of 3D Tensor to 4D Tensor)
        features = torch.stack(features, 0)

        # Merge captions (from tuple of 1D tensor to 2D tensor)
        lengths = [len(tags) for tags in tags_batch]  # List of caption lengths
        #targets_t = torch.zeros(len(tags_batch), max(lengths)).long()
        targets_t = torch.zeros(len(tags_batch), max_seq).long()

        # 単純に各batchが同じ長さになるよう0埋めしている
        for i, seq in enumerate(tags_batch):
            t = seq
            end = lengths[i]
            targets_t[i, :end] = t[:end]

        return features, targets_t, lengths
