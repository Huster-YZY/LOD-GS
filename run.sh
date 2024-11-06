python train.py -s $1 -m $2
python render.py -m $2
python metrics.py -m $2
