#!/bin/bash
python evaluate_cityscapes.py --data-dir ../dataset/Cityscapes/leftImg8bit_trainvaltest/\
                              --save ./results/multi_match_005_0025/\
                              --restore-from ./snapshots/GTA2Cityscapes_multi_005_0025/GTA5_1200.pth