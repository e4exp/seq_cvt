#!/bin/bash

name=052_reformer_nocut_singletext_noweight_resnet50
#target=gt
target=pred

# start hosting
nohup python flask_main.py ${name} ${target} &
#python flask_main.py ${name} ${target} 

# start capturing screenshot
node take_screenshot.js ${name} ${target}