# MODEL_CONFIG=./segnext/models/default/plainvit_base1024_hqseg44k_sax2.py
# torchrun --nproc-per-node=4 \
# 	     --master-port 29504 \
# 	     ./segnext/train.py ${MODEL_CONFIG} \
# 	     --weights ./weights/vitb_sa2_cocolvis_epoch_90.pth \
# 	     --batch-size=12 \
# 	     --gpus=0,1,2,3

# train vitb-sax2 model on coco+lvis
MODEL_CONFIG=./segnext/models/default/plainvit_base1024_cocolvis_sax2_1click.py
export PYTHONPATH=$PYTHONPATH:/home/fmarchesoni/promptSeg
torchrun --nproc-per-node=1 --master-port 29505 ./segnext/train.py ${MODEL_CONFIG} --batch-size=7 --gpus=0 --workers=48 
 