# MODEL_CONFIG=./segnext/models/default/plainvit_base1024_hqseg44k_sax2.py
# torchrun --nproc-per-node=4 \
# 	     --master-port 29504 \
# 	     ./segnext/train.py ${MODEL_CONFIG} \
# 	     --weights ./weights/vitb_sa2_cocolvis_epoch_90.pth \
# 	     --batch-size=12 \
# 	     --gpus=0,1,2,3

# train vitb-sax2 model on coco+lvis
export PYTHONPATH=$PYTHONPATH:/home/fmarchesoni/promptSeg
# # many clicks at 512 resolution
# MODEL_CONFIG=./segnext/models/default/plainvit_base512_cocolvis_sax2.py
# torchrun --nproc-per-node=1 --master-port 29505 ./segnext/train.py ${MODEL_CONFIG} --batch-size=24 --gpus=1 --workers=24
# # 1 click at 512 resolution
# MODEL_CONFIG=./segnext/models/default/plainvit_base512_cocolvis_sax2_1click.py
# torchrun --nproc-per-node=1 --master-port 29515 ./segnext/train.py ${MODEL_CONFIG} --batch-size=24 --gpus=0 --workers=24 
# 1 click at 512 resolution lite augmentations
MODEL_CONFIG=./segnext/models/default/plainvit_base512_cocolvis_sax2_1click_liteaug.py
torchrun --nproc-per-node=1 --master-port 29515 ./segnext/train.py ${MODEL_CONFIG} --batch-size=24 --gpus=1 --workers=24 
# # run 1 click pos neg at 512 resolution
# MODEL_CONFIG=./segnext/models/default/plainvit_base512_cocolvis_sax2_1posneg.py
# torchrun --nproc-per-node=1 --master-port 29506 ./segnext/train.py ${MODEL_CONFIG} --batch-size=24 --gpus=1 --workers=24
## debug
# torchrun --nproc-per-node=1 --master-port 29525 ./segnext/train.py ${MODEL_CONFIG} --batch-size=4 --gpus=0 --workers=0
