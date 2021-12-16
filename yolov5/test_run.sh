clear
python3 detect.py \
	--weights runs/train/exp1/weights/best.pt \
	--img 640 \
	--conf 0.25 \
	--source ../datasets/gtacar/images/test_small
