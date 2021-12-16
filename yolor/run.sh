clear
python3 train_rob.py \
	--batch 1 \
	--epochs 2 \
	--workers 0 \
	--log-imgs 1 \
	--img-size 320 320 \
	--data ../datasets/gtacar_small_yolor.yaml


# python3 train_rob.py \
# 	--batch 1 \
# 	--epochs 2 \
# 	--workers 0 \
# 	--log-imgs 1 \
# 	--img-size 320 320 \
#     --weights runs/train/2021-12-15-06-43-52/weights/best_p.pt \
# 	--data ../datasets/gtacar_small_yolor.yaml
