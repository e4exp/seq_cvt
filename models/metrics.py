from logging import getLogger
from typing import Tuple
logger = getLogger(__name__)

import numpy as np
import torch
from numpy.core.fromnumeric import mean, std
from pytorch_msssim import ssim, ms_ssim
from skimage.metrics import structural_similarity as ssim_sk
from skimage.metrics import mean_squared_error


def ssim_skimage(imgs_gt: np.ndarray, imgs_pred: np.ndarray):
    _mse = mean_squared_error(imgs_gt, imgs_pred)
    _ssim = ssim_sk(imgs_gt,
                    imgs_pred,
                    data_range=imgs_pred.max() - imgs_pred.min())
    return _mse, _ssim


def ssim_average(imgs_gt: list, imgs_pred: list):
    """
    imgs_gt: list of np.ndarray
    imgs_pred: list of np.ndarray
    """

    imgs_gt = np.array(imgs_gt)
    imgs_pred = np.array(imgs_pred)

    # channel first
    imgs_gt = np.transpose(imgs_gt, (0, 3, 1, 2))
    imgs_pred = np.transpose(imgs_pred, (0, 3, 1, 2))

    #convert np.ndarray to torch.tensor
    imgs_pred = torch.from_numpy(imgs_pred.astype(np.float32)).clone()
    imgs_gt = torch.from_numpy(imgs_gt.astype(np.float32)).clone()

    #get ssim/ms-ssim value
    result_ssim = ssim(imgs_pred, imgs_gt, data_range=255, size_average=True)
    result_ms_ssim = ms_ssim(imgs_pred,
                             imgs_gt,
                             data_range=255,
                             size_average=True)

    return result_ssim, result_ms_ssim


def error_exact(htmls_gt, htmls_pred):
    errors = []

    # for test files
    logger.debug(len(htmls_gt))
    logger.debug(len(htmls_pred))

    for i in range(len(htmls_gt[0])):
        html_pred = htmls_pred[i]
        html_gt = htmls_gt[i][0]

        # for html tags
        length = max(len(html_pred), len(html_gt))
        if len(html_pred) == length:
            # pad html_gt
            html_gt = html_gt + ["無効"] * (length - len(html_gt))
        else:
            #pad html_pred
            html_pred = html_pred + ["無効"] * (length - len(html_pred))

        error = 0
        for j, tag in enumerate(html_pred):
            if tag != html_gt[j]:
                error += 1
        error /= length
        errors.append(error)
    error_mean = mean(errors)
    error_std = std(errors)
    return error_mean, error_std


def accuracy_exact(htmls_gt, htmls_pred):
    scores = []

    # for test files
    logger.debug(len(htmls_gt))
    logger.debug(len(htmls_pred))

    for i in range(len(htmls_gt[0])):
        html_pred = htmls_pred[i]
        html_gt = htmls_gt[i][0]

        # for html tags
        length = max(len(html_pred), len(html_gt))
        len_stop = min(len(html_pred), len(html_gt))
        score = 0
        for j, tag in enumerate(html_pred):
            if j >= len_stop:
                break
            if tag == html_gt[j]:
                score += 1
        score /= length
        scores.append(score)

    logger.debug(scores)
    score_mean = mean(scores)
    score_std = std(scores)
    return score_mean, score_std


if __name__ == "__main__":

    import glob
    import os
    import argparse

    import cv2
    from PIL import Image
    from tqdm import tqdm
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms

    class RenderedImageDataSet(Dataset):
        def __init__(self, paths_gt, path_dir_pred):

            self.paths_pred = []
            self.paths_gt = paths_gt
            self.h_max = 0
            self.w_max = 0

            for path_gt in tqdm(self.paths_gt):
                name = os.path.basename(path_gt)
                path_pred = os.path.join(path_dir_pred,
                                         name.replace("gt", "pred"))
                self.paths_pred.append(path_pred)

                # img_pred = Image.open(path_pred).convert('RGB')
                # img_gt = Image.open(path_gt).convert('RGB')

                # w_gt, h_gt = img_gt.size
                # w_pred, h_pred = img_pred.size
                # h = max(h_gt, h_pred)
                # w = max(w_gt, w_pred)
                # if h > self.h_max:
                #     self.h_max = h
                # if w > self.w_max:
                #     self.w_max = w

        def __len__(self):
            return len(self.paths_gt)

        def __getitem__(self, idx):
            path_gt = self.paths_gt[idx]
            path_pred = self.paths_pred[idx]

            # get image
            img_pred = Image.open(path_pred).convert('RGB')
            img_gt = Image.open(path_gt).convert('RGB')

            # make padding
            x = 0
            y = 0
            w_gt, h_gt = img_gt.size
            w_pred, h_pred = img_pred.size
            h = max(h_gt, h_pred)
            w = max(w_gt, w_pred)
            pad_pred = Image.new("RGB", (w, h), (255, 255, 255))
            pad_pred.paste(img_pred, (x, y))
            pad_pred = transforms.ToTensor()(pad_pred)

            pad_gt = Image.new("RGB", (w, h), (255, 255, 255))
            pad_gt.paste(img_gt, (x, y))
            pad_gt = transforms.ToTensor()(pad_gt)

            return pad_pred, pad_gt

    parser = argparse.ArgumentParser()
    parser.add_argument("--path_pred")
    parser.add_argument("--path_gt")
    args = parser.parse_args()

    paths_pred = glob.glob(args.path_pred + "/*.jpg")
    paths_pred += glob.glob(args.path_pred + "/*.png")

    paths_gt = glob.glob(args.path_gt + "/*.jpg")
    paths_gt += glob.glob(args.path_gt + "/*.png")

    ssim_mean = 0
    msssim_mean = 0
    cnt = 0
    bs = 1
    nw = 4

    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = "cpu"

    dataset = RenderedImageDataSet(paths_gt, args.path_pred)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=bs,
        shuffle=False,
        num_workers=nw,
        #pin_memory=True,
    )

    for i, (imgs_pred, imgs_gt) in enumerate(tqdm(dataloader)):

        imgs_pred.to(device)
        imgs_gt.to(device)
        result_ssim = ssim(imgs_pred,
                           imgs_gt,
                           data_range=255,
                           size_average=True)
        result_msssim = ms_ssim(imgs_pred,
                                imgs_gt,
                                data_range=255,
                                size_average=True)
        ssim_mean += result_ssim
        msssim_mean += result_msssim
        cnt += 1

    print("len of imgs: {}".format(cnt))
    print("ssim mean: {}".format(ssim_mean / cnt))
    print("ms-ssim mean: {}".format(msssim_mean / cnt))
