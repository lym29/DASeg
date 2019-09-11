#!/bin/bash
python train_gta2cityscapes_multi.py --snapshot-dir ./snapshots/GTA2Cityscapes_multi_005_0025 \
                                     --lambda-seg 0.1 \
                                     --lambda-adv-target1 0.0002 --lambda-adv-target2 0.001\
                                     --lambda-match-target1 0.05 --lambda-match-target2 0.025\
                                     --tensorboard\
                                     --input-size 640,360\
                                     --input-size-target 512,256\
                                     #--restore-from ./snapshots/GTA2Cityscapes_multi/GTA5_2000.pth
                                     
