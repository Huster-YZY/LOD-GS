python train.py -s $1 -m $2 --eval --white_background
python render.py -m $2 --skip_train --white_background
python metrics.py -m $2
# python train.py -s ~/dataset/multiscale_nerf_synthetic/ship -m ./output/mtmt/ship --eval --load_allres
