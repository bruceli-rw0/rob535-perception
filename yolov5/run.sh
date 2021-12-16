clear
# python3 train_rob.py \
# 	--img 320 \
# 	--batch 1 \
# 	--epochs 1 \
# 	--workers 0 \
# 	--data ../datasets/gtacar_small.yaml

python3 train_rob.py \
	--img 640 \
	--batch -1 \
	--epochs 10 \
	--workers 0 \
	--data ../datasets/gtacar.yaml
