clear
python3 detect.py \
	--weights runs/train/2021-12-15-16-22-50/weights/best.pt \
	--img 640 \
	--conf 0.25 \
	--source ../datasets/gtacar/images/test
