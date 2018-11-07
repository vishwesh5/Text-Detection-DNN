rm crop*
python3 text_detection.py --input input-hd.jpg --model frozen_east_text_detection.pb
cd OCR/
python3 run_images.py
