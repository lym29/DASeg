#!/bin/bash
python evaluate_cityscapes.py --data-dir ../dataset/Cityscapes/leftImg8bit_trainvaltest/\
                              --save ./results/local_00002/\
                              --restore-from ./snapshots/local_00002/GTA5_55000.pth\
                              #--cpu