## dataset download
download to the same folder train and val from lvis page and the annotations from the link in the readme
uncompress all, we end up with dirs: [train2017, val2017, train, val]
in fact x2017 should be inside x/images
so we do
`ln -s $PWD/train2017 $PWD/train/images`
`ln -s $PWD/val2017 $PWD/val/images`


## reproducing results
- using their eval proc
    DAVIS mIoU@1=71.96%;
    HQSeg44K mIoU@1=37.59%;
- with my implementation of a randomclicker using their evaluation procedure
    DAVIS mIoU@1=65.55%;
    HQSeg44K mIoU@1=31.63%;
- with my implementation of a randomclicker using their evaluation procedure and threshold 0.5< 
    DAVIS mIoU@1=65.31%;
    HQSeg44K mIoU@1=31.08%;
- with my eval procedure but their dataloaders  <--- BASELINE 
    DAVIS mIoU@1=65.31%;            
    HQSeg44K mIoU@1=31.08%;

## model takes a lot to get to good performance 
- with my eval and their train model with 1 click (epoch 19)
    DAVIS mIoU@1=56.29%;
    HQSeg44K mIoU21=20.96%;
it's considerably worse for now. More training? or do more clicks help? strange
- with my eval and their train model with 1 click (epoch 29)
    DAVIS mIoU@1=61.22%;
    HQSeg44K mIoU@1=25.06%;
well it's getting there, -4 for davis and -6 for HQSeg44k
- with my eval and their train model with 1 click (epoch 64)
    DAVIS mIoU@1=62.60%;  -2.71
    HQSeg44K mIoU@1=26.87%;  -4.21
- with my eval and their train model with 1 click (epoch 84)
    DAVIS mIoU@1=63.73%;  -1.58
    HQSeg44K mIoU@1=29.23%;  -1.85
- with my eval and their train model with 1 click (epoch 90)
    DAVIS mIoU@1=64.29%;  -1.02
    HQSeg44K mIoU@1=28.78%;  -2.26
- with my eval and their train model with 1 click (epoch 94+)
    DAVIS mIoU@1=64.29%;  -1.02
    HQSeg44K mIoU@1=29.04%;  -2.04

## how to run
- train: `bash run_train.sh`
- evaluate: `python segnext/scripts/my_evaluate_model.py model_mmdd_yyyy/default/plainvit_base1024_cocolvis_sax2/019/checkpoints/090.pth`. Our run is 019

## our incremental experiments
- one positive click only full resolution (021)
- one positive click only but 512 img size (022)
    - comment: 512 seems to perform better than the old experiment on full resolution (019), and trains faster, which is great news. However, I'm not sure the old experiment (019) implemented one-click correctly, I am more confident of (021). In any case, for sure 512 + single click improves over the baseline segnext, which is great news!
- one click (positive OR negative) at 512 res (023)
    - comment: seems to perform better than the one with a single positive click, which is great news 
- comment on strange things:
    - sometimes there's no mask for the image-mask pair
    - davis perf depends a lot on the clicks (need to sample many)
    - the clicks i was doing didn't look random

## geometric augmentations don't improve performance
- python segnext/scripts/my_evaluate_model.py model_mmdd_yyyy/default/plainvit_base1024_cocolvis_sax2/022/checkpoints/last_checkpoint.pth --c=5
    mean iou for DAVIS 0.7100395402508232                                               
    mean iou for HQSeg44K 0.34390783530958574                                           
- python segnext/scripts/my_evaluate_model.py model_mmdd_yyyy/default/plainvit_base1024_cocolvis_sax2/022/checkpoints/last_checkpoint.pth --aug --c=5
    mean iou for DAVIS 0.6963395196501662                                               
    mean iou for HQSeg44K 0.33504551363956475
- why? because the models are not equivariant to them (for instance vertical flips)

## my model training failed (for now)
- with my eval and my trained model
    seems that the model doesn't train
    DAVIS mIoU@1=14.94%;


## next
- wait for 021 to finish
    - compare 021 vs. 022.
- wait for 023 to finish
    - compare 022 vs. 023.
    - implement multi-click inference time scaling
