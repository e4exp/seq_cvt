import os
import random
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
        self.is_train = flg_make_vocab

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

        # make test indices
        # indices_test = []
        # for i in range(len(self.paths_image)):
        #     if random.random() > 0.5:
        #         indices_test.append(i)
        #     else:
        #         while True:
        #             idx_fake = random.randint(0, self.__len__() - 1)
        #             tags = self.htmls[i]
        #             tags_new = self.htmls[idx_fake]

        #             if tags != tags_new:
        #                 indices_test.append(idx_fake)
        #                 break

        #self.indices_test = indices_test
        #print("indices_test", indices_test)
        self.indices_test = [
            1353, 673, 527, 1428, 33, 5, 1567, 7, 8, 9, 10, 11, 106, 13, 623,
            1084, 16, 17, 612, 826, 20, 132, 22, 23, 1515, 351, 1573, 27, 28,
            1304, 991, 1327, 1707, 33, 34, 35, 36, 1196, 624, 39, 1801, 1668,
            42, 100, 469, 45, 630, 47, 995, 1011, 1709, 183, 52, 215, 238,
            1622, 56, 333, 58, 59, 1662, 61, 62, 1156, 849, 1424, 715, 67, 68,
            1660, 70, 1579, 72, 73, 378, 1948, 76, 77, 78, 79, 1047, 81, 1378,
            1546, 504, 1977, 86, 624, 288, 268, 90, 91, 609, 1241, 1006, 1520,
            96, 97, 1084, 909, 100, 1777, 102, 103, 104, 827, 1519, 1672, 547,
            1079, 923, 313, 872, 113, 221, 1774, 1777, 86, 567, 119, 1784, 121,
            1953, 123, 124, 125, 1392, 127, 322, 129, 130, 131, 1410, 133, 134,
            1064, 136, 1449, 138, 139, 1698, 141, 353, 143, 1708, 1365, 146,
            147, 66, 149, 1058, 1586, 152, 281, 674, 155, 156, 1554, 20, 159,
            160, 161, 116, 706, 596, 527, 166, 167, 168, 1134, 759, 1055, 172,
            1061, 924, 175, 891, 177, 178, 179, 180, 224, 1344, 556, 184, 685,
            1448, 1295, 1325, 189, 190, 191, 1673, 1229, 1882, 1573, 597, 197,
            198, 199, 200, 201, 202, 1023, 204, 479, 206, 1658, 1177, 209, 377,
            1079, 212, 747, 563, 215, 808, 217, 218, 1980, 1477, 221, 222, 223,
            224, 225, 500, 553, 228, 255, 1418, 231, 129, 233, 1242, 235, 236,
            888, 238, 1365, 240, 1416, 969, 1494, 318, 245, 246, 247, 1952,
            249, 250, 251, 1783, 253, 254, 255, 256, 718, 258, 1693, 1104, 193,
            529, 263, 1054, 89, 1719, 372, 1192, 672, 270, 271, 1701, 1157,
            274, 1303, 276, 745, 1881, 231, 716, 677, 1359, 283, 376, 285, 428,
            287, 1675, 135, 700, 463, 292, 731, 1620, 542, 509, 248, 1156, 299,
            1502, 1944, 302, 859, 1483, 305, 306, 1485, 1365, 309, 310, 311,
            679, 313, 1057, 1571, 101, 1087, 478, 319, 320, 321, 322, 1667,
            1376, 514, 1969, 327, 328, 329, 330, 1645, 837, 852, 782, 1858,
            336, 337, 314, 451, 340, 341, 342, 343, 1799, 1035, 194, 347, 348,
            349, 755, 351, 352, 419, 991, 355, 1062, 357, 358, 359, 975, 239,
            362, 363, 1915, 365, 1607, 948, 1282, 42, 370, 508, 1794, 373, 810,
            865, 896, 1622, 378, 379, 380, 170, 382, 383, 384, 775, 386, 998,
            1259, 1264, 1772, 636, 465, 332, 394, 315, 396, 1812, 547, 1050,
            906, 401, 1541, 648, 404, 525, 669, 264, 408, 409, 410, 411, 262,
            413, 552, 415, 617, 417, 351, 419, 42, 421, 422, 1215, 424, 90,
            426, 1857, 851, 429, 430, 1624, 714, 433, 817, 672, 298, 437, 438,
            1727, 440, 441, 1408, 443, 444, 445, 528, 1808, 448, 449, 858, 623,
            667, 341, 781, 455, 1570, 1719, 1095, 459, 460, 461, 462, 463,
            1501, 465, 1898, 411, 468, 1899, 1907, 471, 472, 473, 474, 193,
            476, 1201, 478, 633, 446, 245, 426, 483, 484, 485, 486, 487, 488,
            489, 490, 491, 492, 1370, 1258, 54, 1240, 497, 473, 739, 500, 1236,
            1847, 503, 1842, 1771, 241, 507, 508, 336, 510, 511, 183, 513, 514,
            802, 1211, 517, 1288, 97, 1121, 409, 1003, 523, 524, 525, 526, 527,
            1853, 1859, 394, 898, 1605, 1329, 1479, 731, 536, 537, 538, 539,
            1621, 541, 30, 575, 789, 545, 990, 1652, 1137, 1038, 1465, 249,
            552, 553, 1278, 1171, 556, 153, 558, 559, 234, 1473, 1079, 1937,
            1008, 1644, 266, 567, 1285, 569, 1094, 571, 572, 1260, 574, 679,
            681, 577, 1791, 579, 594, 1738, 582, 583, 1254, 585, 586, 676, 522,
            1152, 455, 591, 722, 1554, 457, 1742, 596, 568, 606, 508, 1174,
            601, 602, 1399, 837, 1754, 606, 175, 1734, 1152, 610, 898, 1307,
            613, 1254, 412, 616, 1755, 618, 619, 620, 621, 1515, 623, 624, 625,
            983, 627, 841, 629, 630, 631, 594, 633, 634, 879, 636, 1066, 1321,
            639, 640, 1488, 642, 729, 644, 645, 646, 49, 648, 712, 516, 651,
            1337, 653, 1963, 1366, 637, 1000, 658, 1490, 660, 1516, 662, 866,
            691, 665, 113, 1266, 668, 669, 961, 671, 672, 1972, 674, 675, 676,
            1611, 905, 1650, 680, 264, 682, 781, 684, 685, 193, 1230, 819,
            1390, 690, 51, 1306, 693, 694, 936, 305, 697, 1808, 1160, 918, 701,
            702, 703, 704, 705, 706, 707, 1159, 709, 710, 711, 1580, 1560,
            1403, 715, 1510, 717, 718, 719, 1578, 1100, 722, 723, 724, 1803,
            726, 1849, 675, 729, 730, 56, 732, 172, 217, 806, 736, 737, 738,
            739, 740, 833, 742, 64, 1656, 745, 1313, 883, 959, 749, 750, 1369,
            752, 753, 1060, 755, 756, 779, 758, 652, 1932, 1416, 1806, 763,
            1596, 765, 766, 767, 1835, 769, 432, 977, 770, 773, 774, 775, 436,
            1716, 778, 779, 742, 781, 782, 1955, 862, 785, 218, 787, 1084, 789,
            1847, 1126, 792, 793, 794, 1820, 796, 588, 798, 1849, 960, 1107,
            802, 803, 804, 1152, 425, 807, 808, 1596, 161, 608, 812, 813, 814,
            815, 592, 817, 228, 819, 820, 821, 822, 823, 943, 825, 302, 1167,
            828, 234, 1955, 831, 832, 1864, 1614, 835, 836, 837, 858, 839,
            1374, 1694, 312, 843, 691, 854, 846, 1644, 1962, 849, 850, 1337,
            852, 853, 854, 855, 856, 857, 932, 859, 236, 861, 1415, 863, 864,
            865, 275, 1593, 868, 87, 870, 400, 1141, 873, 1217, 1918, 876,
            1674, 878, 879, 880, 881, 882, 1646, 884, 885, 886, 1702, 888, 889,
            890, 18, 892, 45, 399, 895, 1629, 890, 1280, 143, 900, 306, 902,
            266, 904, 737, 694, 907, 696, 1132, 910, 1161, 1016, 913, 1485,
            990, 1789, 480, 458, 919, 85, 1873, 922, 923, 1712, 585, 926, 620,
            1185, 929, 1581, 1607, 237, 1828, 934, 1451, 1661, 422, 700, 976,
            1343, 941, 942, 943, 1006, 1940, 946, 947, 522, 949, 904, 1736,
            952, 313, 954, 955, 1236, 957, 958, 959, 960, 961, 1014, 963, 234,
            965, 966, 967, 968, 969, 970, 747, 650, 1812, 974, 543, 976, 977,
            978, 979, 980, 981, 64, 901, 529, 985, 986, 987, 988, 17, 1649,
            991, 992, 686, 994, 1462, 996, 170, 998, 1391, 1000, 1001, 1002,
            1917, 1962, 1330, 1654, 1777, 195, 1356, 643, 1011, 1012, 516,
            1014, 166, 1016, 914, 558, 1553, 1020, 1906, 1022, 545, 1014, 1025,
            874, 1027, 1820, 789, 1030, 329, 1032, 1033, 514, 919, 1402, 1037,
            1038, 1541, 1040, 1308, 1042, 344, 1044, 1045, 1046, 1047, 799,
            1520, 1087, 1051, 1052, 1268, 726, 8, 586, 1817, 1859, 363, 1136,
            1061, 1062, 526, 1064, 1608, 1066, 1067, 240, 427, 1070, 1071, 468,
            1073, 410, 1075, 176, 1077, 1078, 1079, 1080, 450, 1082, 1083,
            1084, 1085, 1086, 1087, 1919, 1089, 1677, 765, 1092, 1093, 1607,
            1095, 1096, 1097, 1098, 528, 1100, 1951, 1421, 1103, 1104, 162,
            801, 279, 1108, 271, 1110, 1229, 432, 1113, 1114, 1249, 1116, 1582,
            1118, 1119, 1120, 1477, 971, 640, 1124, 734, 1694, 1127, 1128,
            1129, 1130, 1131, 1132, 682, 1134, 1135, 900, 1137, 996, 498, 1140,
            1141, 1142, 1034, 187, 1145, 1146, 1147, 155, 551, 1150, 422, 1874,
            133, 1011, 459, 1179, 1858, 1171, 1159, 1160, 812, 1162, 334, 1164,
            1165, 1322, 1871, 1888, 1169, 1617, 1829, 1345, 1173, 1174, 206,
            1889, 1177, 1178, 1179, 1180, 1181, 1182, 1832, 97, 1185, 1186,
            1187, 1188, 1348, 1190, 1191, 1192, 1646, 1194, 82, 1196, 1197,
            1198, 1502, 178, 1201, 995, 1203, 347, 33, 542, 742, 1208, 1209,
            552, 774, 1212, 221, 1911, 1232, 1062, 1126, 749, 1219, 627, 1312,
            1299, 1223, 1224, 1225, 1226, 1227, 1228, 194, 1230, 1473, 723,
            1233, 1234, 1637, 893, 952, 1238, 150, 1945, 677, 1242, 446, 1033,
            1398, 1096, 647, 615, 1325, 831, 1251, 1252, 1253, 1859, 1255,
            1536, 309, 1258, 1259, 1260, 1261, 1262, 1263, 1264, 1265, 762,
            1874, 301, 205, 1270, 1271, 1272, 1273, 1274, 1275, 1284, 328,
            1278, 262, 1280, 625, 1282, 1283, 1284, 1285, 1375, 1287, 223,
            1289, 1039, 108, 1042, 1293, 1294, 938, 1670, 1814, 420, 622, 1949,
            1301, 1830, 200, 1148, 1305, 1306, 126, 1308, 1342, 1310, 1311,
            1312, 1061, 1314, 1315, 1316, 1868, 790, 1311, 1320, 1321, 438,
            822, 648, 236, 1326, 562, 1328, 1329, 617, 568, 886, 1333, 1334,
            1335, 460, 1337, 1338, 255, 1500, 1341, 1342, 1343, 1344, 1783,
            529, 340, 384, 1349, 1895, 396, 667, 1353, 1354, 1166, 1637, 1357,
            1358, 1359, 1360, 1798, 443, 1363, 409, 1365, 1405, 1725, 1368,
            1468, 1370, 1371, 1915, 1373, 856, 1422, 614, 1377, 65, 1379, 1285,
            84, 1208, 1331, 1384, 1385, 1630, 1850, 1388, 1770, 212, 1391,
            1392, 1393, 1394, 1935, 1396, 1075, 1398, 1399, 1641, 1020, 665,
            243, 1404, 294, 1406, 1407, 386, 1169, 1410, 703, 1412, 1413, 1920,
            1415, 1416, 1417, 1054, 638, 1420, 1421, 1422, 1423, 956, 1425,
            1469, 1659, 1428, 1429, 1430, 1346, 1432, 1433, 1961, 1435, 1306,
            1437, 919, 1943, 1173, 1441, 1318, 1443, 1444, 773, 1446, 684,
            1448, 1449, 1531, 830, 510, 1453, 387, 1455, 1456, 378, 1858, 1459,
            283, 1238, 1462, 841, 1539, 1465, 1466, 1467, 215, 1443, 1457,
            1471, 602, 1791, 1474, 1475, 1476, 1027, 1478, 1733, 1480, 1481,
            1482, 1483, 1053, 1485, 1486, 1487, 1488, 1833, 1490, 1491, 1138,
            913, 1494, 1495, 1496, 1497, 555, 824, 1867, 1501, 1007, 537, 584,
            1505, 1506, 311, 1508, 1509, 337, 1511, 1832, 1588, 1514, 915, 86,
            1517, 1701, 1519, 1918, 374, 755, 1431, 1524, 1212, 1526, 1871,
            1528, 1003, 1530, 1531, 937, 1533, 1534, 1535, 1536, 1537, 907,
            1444, 1540, 1163, 1542, 1543, 1544, 1545, 1546, 1547, 1548, 1066,
            1550, 1047, 1552, 1839, 1554, 1555, 8, 1780, 1558, 1559, 1037,
            1063, 1562, 1563, 1564, 1565, 828, 1283, 1568, 1569, 1570, 1571,
            1572, 1573, 778, 1575, 1131, 1634, 1578, 135, 1580, 1581, 1882,
            1506, 237, 1585, 215, 1587, 1899, 1589, 42, 1591, 1398, 1700, 852,
            241, 1596, 1422, 1598, 1599, 330, 1601, 239, 1603, 225, 1537, 1606,
            695, 697, 1609, 511, 1611, 1612, 1613, 1721, 859, 1616, 1617, 1618,
            1619, 1620, 1621, 1527, 1623, 954, 1625, 1827, 1627, 943, 1444,
            1499, 511, 535, 1633, 1087, 1470, 1045, 789, 849, 1626, 691, 1826,
            1642, 1548, 1644, 1645, 1646, 1647, 1648, 1649, 928, 285, 73, 1653,
            896, 1655, 1656, 1657, 171, 1627, 620, 1661, 1662, 70, 1834, 1665,
            1666, 461, 1868, 996, 816, 1469, 1672, 1673, 762, 1675, 471, 863,
            1273, 0, 1680, 785, 1887, 1630, 1684, 130, 1686, 1687, 1688, 1689,
            1690, 1691, 1692, 12, 1707, 1790, 1696, 1405, 507, 1699, 1595,
            1701, 1051, 520, 1704, 1705, 1557, 515, 1708, 429, 734, 1711, 1855,
            94, 1160, 1715, 1416, 1717, 1718, 224, 664, 1783, 574, 855, 972,
            1725, 1726, 1727, 1728, 1776, 922, 1837, 1732, 819, 1734, 1735,
            311, 1737, 1738, 1739, 1740, 744, 1427, 1743, 1716, 1562, 853,
            1747, 559, 603, 1750, 382, 1781, 1061, 700, 1755, 352, 821, 1758,
            730, 557, 1761, 1762, 1267, 1764, 1765, 1812, 911, 753, 936, 1770,
            1771, 1772, 471, 1259, 1805, 1776, 1777, 1871, 1779, 1780, 1781,
            1782, 220, 1450, 380, 1786, 1787, 643, 1789, 1724, 1791, 1792,
            1793, 1794, 1795, 1796, 1797, 1798, 1799, 1800, 701, 1802, 1803,
            1804, 1805, 512, 1003, 1792, 1809, 263, 1783, 1812, 1179, 617,
            1815, 1610, 457, 685, 811, 1820, 1821, 67, 1823, 1356, 116, 56,
            1827, 1402, 1970, 824, 1831, 1424, 1833, 951, 1835, 1836, 1709,
            1838, 33, 1840, 1509, 1842, 291, 1844, 1845, 1846, 275, 377, 1849,
            838, 1851, 140, 1853, 1854, 1855, 1856, 1968, 972, 1977, 1860, 274,
            1588, 1863, 1864, 255, 103, 652, 734, 159, 833, 592, 1872, 1873,
            1442, 869, 742, 1877, 1878, 189, 1880, 1881, 307, 1883, 1053, 35,
            608, 1887, 1888, 1889, 1890, 1891, 1617, 1893, 1894, 1895, 758,
            627, 1898, 1899, 1900, 1901, 703, 1903, 1904, 968, 1906, 1907,
            1908, 539, 519, 1306, 220, 1196, 664, 1915, 1916, 1917, 373, 1919,
            13, 861, 1922, 1086, 1924, 1173, 1926, 1927, 1928, 1929, 504, 1931,
            419, 1430, 1011, 1935, 1355, 784, 1938, 199, 1940, 1941, 1075,
            1433, 1944, 1599, 1505, 858, 1948, 1949, 1950, 84, 554, 1696, 54,
            1955, 1956, 1957, 1951, 1959, 1960, 94, 1962, 1392, 1964, 1905,
            1966, 1548, 1968, 1969, 558, 1971, 690, 1973, 1934, 676, 318, 1977,
            233, 1979, 1980, 46, 1982
        ]

    def __len__(self):
        return len(self.paths_image)

    def __getitem__(self, idx):
        path_img = self.paths_image[idx]
        tags = self.htmls[idx]

        if self.is_train:
            # TODO: make match score
            if random.random() > 0.5:
                # case real
                score_match = 0.8
            else:
                # case fake
                score_match = 0.2
                while True:
                    idx_fake = random.randint(0, self.__len__() - 1)
                    tags_new = self.htmls[idx_fake]

                    if tags != tags_new:
                        tags = tags_new
                        break
        else:
            idx_fake = self.indices_test[idx]
            tags = self.htmls[idx_fake]
            if idx == idx_fake:
                score_match = 0.8
            else:
                score_match = 0.2

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

        # score
        score_match = torch.Tensor(torch.ones(1) * score_match)

        # file name
        idx = torch.Tensor(torch.ones(1) * idx)

        return feature, tags, score_match, idx


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
    #max_seq = 128

    # Sort datalist by caption length; descending order
    data.sort(key=lambda data_pair: len(data_pair[1]), reverse=True)
    features, tags_batch, scores, indices = zip(*data)

    # Merge images (from tuple of 3D Tensor to 4D Tensor)
    features = torch.stack(features, 0)
    scores = torch.stack(scores, 0)
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

    return features, targets_t, lengths, scores, indices
