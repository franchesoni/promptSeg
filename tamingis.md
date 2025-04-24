# Taming Image Segmentation

## Summary
Defining image segmentation is hard, but it boils down to proposing good masks, where ``good'' means ``similar to the ground truth masks''. The usual metric in the community is mAR, which is equivalent to mIoU (we show that). Also, interactive image segmentation methods are evaluated using mIoU@1, and we show \fmc{(to be done)} that mIoU@1 correlates well with mIoU (the first is interactive, the second is zero-shot). This is natural because zero-shot segmentation is based on simulating many individual clicks that are used as inputs of an interactive image segmentor. 
One issue of the evaluation of interactive image segmentation methods is that it frequently overlooks zero-shot image segmentation metrics (such as mIoU), which makes it hard to assess its applicability to zero-shot segmentation. Moreover, the evaluation usually only involves object-centric datasets, where the prominent objects are the ones annotated, which fail to describe the performance when all parts of an image are potentially relevant, for instance in remote sensing, medical or geological images. 
In order to have a better evaluation of zero-shot segmentation we propose to evaluate in HyperSim \fmc{cite}, which has perfect annotations covering almost the whole image. The zero-shot segmentation evaluations usually do not restrict the number of proposals, which has an important effect on the performance, therefore we look at the mIoU vs. number of proposals curve, which provides a better understanding.
To improve zero-shot segmentation, we start from the SegNext architecture, a state-of-the-art interactive image segmentor, and train it for the one-click case (reducing resolution, as it doesn't hurt metrics and improves computational efficiency). In order to allow for interactive annotations we train the network to deal with either one positive click or one negative click. Combining individual predictions by maximum certainty allows for local editing and mask correction without complex methods such as FocalClick.
Inspired by previous segmentation literature we modify the loss to better match the IoU objective, using a differentiable version of the IoU \fmc{cite lovasz}. 
By synthetizing a grid of clicks we obtain our zero-shot segmentation result, which refine by expressing it as a union of superpixels obtained from a processing of the predictions based on connectivity and maximum-certainty. We also replace the box non-maximum-supression of SAM with mask non-maximum-supression, which avoids the biases of bounding boxes. 
Segmentation is multiscale in nature, and in order to train with datasets where masks of different granularities are provided, we propose a loss that makes the model predict level lines correctly. Finally, we provide many visualizations on texture segmentation mosaics, and graph-based methods to obtain partitions of an image in an arbitrary number of regions.
The result is an easy-to-train method that is competitive with the state-of-the-art (SAM2.1, SegNext), for both zero-shot image segmentation, interactive segmentation and mask editing. Finally, we distill our zero-shot image segmentor into a non-interactive one which is much faster and provides a pixel-level feature space, where features that are close to each other correspond to the same object in the image. 
We also report a negative result: data augmentation doesn't work, including pseudo labeling, etc.

## Steps

### Benchmarks
We benchmark interactive methods on their interactive mIoU@1 and zero-shot mIoU capabilities on three datasets with high quality masks: DAVIS (object centric, video), HQSeg44k (object centric, fine structures), and HyperSim (whole image, synthetic). We always use ViT base as a backbone and measure the IoU in the original image resolution. We do not evaluate against masks smaller than 16x16 pixels. 

**Results:**
(run with `python segnext/scripts/my_evaluate_model.py checkpoint_path`)
- For reference, the results of Order-Aware IIS are (mIoU@1center): DAVIS=87.29\% @ 1024, DAVIS=88.05\% @ 2048, HQSeg44K=89.40\% @ 1024, HQSeg44K=89.57\% @ 2048. Note that Order-Aware IIS 1. has no code nor weights, 2. was finetuned on HQSeg44K for 15 epochs.  
- Evaluate official SegNext COCOLVIS-ft-hq44k using official evaluation script (mIoU@1center): DAVIS=85.97\%, HQSeg44K=81.79\% 
<!-- the following are invalid due to the bad random clicker: -->
<!-- - Evaluate official SegNext COCOLVIS-ft-hq44k using official evaluation script but random clicker (mIoU@1): DAVIS=83.06\%, HQSeg44K=80.75\%  -->
<!-- - Evaluate official SegNext COCOLVIS-ft-hq44k using official evaluation script but random clicker and threshold 0.5 (instead of 0.49) (mIoU@1): DAVIS=83.74\%, HQSeg44K=80.57\%  -->
- Evaluate official SegNext COCOLVIS @ epoch 90 using official evaluation script (mIoU@1center): DAVIS=71.96\%, HQSeg44K=64.74\% . From this point we can see that finetuning one epoch on HQSeg44K improves the performance quite a bit. 
- Evaluate official SegNext COCOLVIS @ epoch 90 using official evaluation script but random clicker (mIoU@1): DAVIS=67.75\%, HQSeg44K=59.68\%.  
- Evaluate official SegNext COCOLVIS @ epoch 90 using official evaluation script but random clicker and threshold 0.5 (instead of 0.49) (mIoU@1): DAVIS=67.46\%, HQSeg44K=59.21\%. 
- Evaluate official SegNext COCOLVIS @ epoch 90 using our evaluation script (random and 0.5 thresh) (mIoU@1): DAVIS=67.46\%, HQSeg44K=59.21\%, Hypersim=32.01\%
- Evaluate SAM2.1b+ official weights with our evaluation script: DAVIS=53.15\%, HQSeg44K=47.05\%, Hypersim=42.36\%
- Evaluate repro SegNext COCOLVIS @ epoch 90 res 512 using our evaluation script (random and 0.5 thresh) (mIoU@1): DAVIS=68.76\%, HQSeg44K=58.68\%, Hypersim=26.12\% (85)
- Evaluate repro SegNext COCOLVIS @ epoch 90 res 512 1 click using our evaluation script (random and 0.5 thresh) (mIoU@1): DAVIS=70.26\%, HQSeg44K=60.17\%, Hypersim=24.59\% (86)
- Evaluate repro SegNext COCOLVIS @ epoch 99 res 512 1 click using our evaluation script (random and 0.5 thresh) (mIoU@1): DAVIS=70.57\%, HQSeg44K=60.85\%, Hypersim=24.62\% (86)
- Evaluate repro SegNext COCOLVIS @ epoch 99 res 512 1 click lite aug using our evaluation script (mIoU@1): DAVIS=70.36\%, HQSeg44K=61.70\%, Hypersim=25.10\% (87) (evaluating in tmux 6)
- custom repro v0.0 (evaluating in tmux 7) DAVIS=34.71\%,HQSeg44K=30.24\%,Hypersim=14.77\%

| Method / Setting                                                                                    | DAVIS (%) | HQSeg44K (%) | Hypersim (%) | Notes                                                 |
| --------------------------------------------------------------------------------------------------- | --------- | ------------ | ------------ | ----------------------------------------------------- |
| Order-Aware IIS (mIoU@1center) @ 1024                                                               | 87.29     | 89.40        | -            | No code/weights, finetuned on HQSeg44K for 15 epochs  |
| Order-Aware IIS (mIoU@1center) @ 2048                                                               | 88.05     | 89.57        | -            |                                                       |
| SegNext COCOLVIS-ft-hq44k (official eval, mIoU@1center)                                             | 85.97     | 81.79        | -            |                                                       |
| SegNext COCOLVIS @ epoch 90 (official eval, mIoU@1center)                                           | 71.96     | 64.74        | -            | Finetuning one epoch on HQSeg44K improves performance |
| SegNext COCOLVIS @ epoch 90 (official eval, random clicker, mIoU@1)                                 | 67.75     | 59.68        | -            |                                                       |
| SegNext COCOLVIS @ epoch 90 (official eval, random clicker, threshold 0.5, mIoU@1)                  | 67.46     | 59.21        | -            |                                                       |
| SegNext COCOLVIS @ epoch 90 (our eval, random clicker, threshold 0.5, mIoU@1)                       | 67.46     | 59.21        | 32.01        |                                                       |
| SAM2.1b+ (official weights, our eval, mIoU@1)                                                       | 53.15     | 47.05        | 42.36        |                                                       |
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
| Repro SegNext COCOLVIS @ epoch 90 res 512 (our eval, random clicker, threshold 0.5, mIoU@1)         | 68.76     | 58.68        | 26.12        | (85)                                                  |
| Repro SegNext COCOLVIS @ epoch 90 res 512 1 click (our eval, random clicker, threshold 0.5, mIoU@1) | 70.26     | 60.17        | 24.59        | (86)                                                  |
| Repro SegNext COCOLVIS @ epoch 99 res 512 1 click (our eval, random clicker, threshold 0.5, mIoU@1) | 70.57     | 60.85        | 24.62        | (86)                                                  |
| Repro SegNext COCOLVIS @ epoch 99 res 512 1 click lite aug (our eval, mIoU@1)                       | 70.36     | 61.70        | 25.10        | (87), evaluating in tmux 6                            |
| Custom repro (same than above)                                                                      | 70.46     | 61.55        | 24.91        |                                                       |
| Custom repro @ epoch 10                                                                             | 64.71     | 57.30        | 20.92        |                                                       |
| Custom repro @ epoch 10 posneg                                                                      | 65.63     | 53.28        | 19.03        |                                                       |
| Custom repro @ epoch 10 posneg lovasz                                                               | 63.24     | 47.84        | 08.54        |                                                       |
| Custom repro @ epoch 20 posneg                                                                      | 67.30     | 55.37        | 21.40        |                                                       |
| Custom repro @ epoch 20 posneg lovasz                                                               | 65.87     | 52.75        | 16.66        |                                                       |
| Custom repro @ epoch 70                                                                             | 69.68     | 61.95        | 24.80        |                                                       |
| Custom repro @ epoch 70 posneg                                                                      | 68.78     | 59.75        | 24.40         |                                                       |
| Custom repro @ epoch 70 posneg lovasz                                                               | 70.29     | 62.31        | 20.71        |                                                       |
| Custom repro @ epoch 100 posneg lovasz                                                              | 70.52     | 62.31        | 20.71        |                                                       |
| Custom repro @ epoch 160 posneg lovasz                                                              | 70.65     | 60.58        | 22.11        |                                                       |
| Custom repro @ epoch 130 posneg                                                                     | 69.30     | 59.99        | 24.36        |                                                       |

# thought, next 
thought: lovasz seems to help on object centric but hurt for hypersim. I wonder if it's because getting better in one means being worse in the other.

evaluating our repro over checkpoitns we see that with 10 epochs we get 65 davis and with 80 epochs we get 70 davis (70.51 at 90 70.46 at 99), so we can use 10 epochs for finding our best designs.

conclusions:
- order aware claims to be the best
- hqseg finetuning helps
- random click reduces perf
- changing threshold reduces perf
- our evaluation script preserves perf
- sam2.1+ loses on davis,hqseg but wins on hypersim
- reducing resolution sustains performance in davis,hqseg but lowers performance -6 pp in hypersim
- 1 click training increases performance in davis,hqseg but lowers -2 pp in hypersim
- epoch 99 improves over epoch 90


# Zero-shot segmentation evaluation
We can assert that
1. image segmentation should be evaluated by measuring the ability to recover ground truth masks,
2. the mIoU (or mAR) is the metric of choice, and
3. we need to look at the mIoU vs. number of masks curve.

Therefore a zero-shot segmentation method is such that, given an image, returns an ordered sequence of masks.
Given such sequence, we can compute the mIoU-tries curve, and also optimize the sequence in an oracular way (with a greedy method) and get mIoU-oracle tries. This second curve might be useful to see if the mask ordering method can be improved.

## Next steps
we still need to:
~1. evaluate sam2.1 on the same datasets~
~2. evaluate all methods over hypersim~
    - download hypersim `python download_hypersim.py --contains images/scene_cam_00_final_preview/frame.0000.color.jpg -d /export/home/data/hypersim/ -o` and `python download_hypersim.py --contains images/scene_cam_00_geometry_hdf5/frame.0000.render_entity_id.hdf5 -d /export/home/data/hypersim/`, had to download by hand (curl+unzip+move) the render_identity_id for ai_004_009 (the download consistently failed)
    - create dataset for hypersim
    - running eval (takes 1.5 hs) for original_cocolvis@epoch90 for hypersim (need to run for sam2.1b+ too)
3. evaluate all methods over all datasets in the zero-shot setting for variable number of masks
4. do the scatter, per dataset, where one image is one point, of the miou@0 vs. miou@1
improvements in mind:
- lovasz loss
- track iou too
- multiscale loss
- scheduled cadamw / sing
- make resolution independent
- train with sa1b
- scale

current:
- tmux 0 is running 1 click 512 lite aug
- tmux 1 is running 1 click 512 lite aug repro (batch-size=24)
plan: 
- evaluate zero-shot?
- compare repro vs original (very bad)
- build from there (-valdataset, +aug, +scheduled cadamw, +lovasz)

# Appendix

## Data
We download following the links in the readme of uncbiag/SegNext DAVIS, HQSeg44K, and cocolvis, and extract all files. 

For cocolvis directory, we put the compressed files in the same folder, extract them, and we end up with dirs: [train2017, val2017, train, val]. Now we require x2017 to be inside x/images, which we achieve with
`ln -s $PWD/train2017 $PWD/train/images`
`ln -s $PWD/val2017 $PWD/val/images`

Of course, the paths in the `config.yml`should be updated accordingly


