#!/bin/bash
#rm -rf ~/DASeg/log/*
python train_gta2cityscapes_multi.py --snapshot-dir ./snapshots/local_00002 \
                                     --lambda-seg 0.1 \
                                     --lambda-adv-target1 0.0002 --lambda-adv-target 0.001\
                                     --lambda-adv-local 0.0002\
                                     --lambda-match-target1 0.025 --lambda-match-target2 0.05\
                                     --tensorboard\
                                     --input-size 1024,512\
                                     --input-size-target 1024,512\
                                     #--restore-from ./snapshots/local_00002/GTA5_5000.pth
                                     
