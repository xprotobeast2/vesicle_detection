git clone https://github.com/eriklindernoren/PyTorch-YOLOv3
( cd PyTorch-YOLOv3/ ; sudo pip3 install -r requirements.txt )
( cd PyTorch-YOLOv3/weights/ ; bash download_weights.sh )
touch PyTorch-YOLOv3/__init__.py
mkdir yolov3
mv PyTorch-YOLOv3/* yolov3/
rm -rf PyTorch-YOLOv3
mv yolov3/utils/parse_config.py utils/parse_config.py
mv yolov3/utils/utils.py utils/utils.py
