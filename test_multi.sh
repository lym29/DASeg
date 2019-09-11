#!/bin/bash
python evaluate_cityscapes.py --data-dir ../dataset/Cityscapes/leftImg8bit_trainvaltest/\
                              --save ./results/multi_match_00002_00001/\
                              --restore-from ./snapshots/GTA2Cityscapes_multi_00002_00001/GTA5_6700.pth