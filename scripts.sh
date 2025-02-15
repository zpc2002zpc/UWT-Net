dataset=busi
input_size=256
python train.py  --dataset ${dataset} --input_w ${input_size} --input_h ${input_size} --name ${dataset}_UWTNET  --data_dir [YOUR_DATA_DIR]
python val.py --name ${dataset}_UWTNET--output_dir

dataset=glas
input_size=512
python train.py --dataset ${dataset} --input_w ${input_size} --input_h ${input_size} --name ${dataset}_UWTNET  --data_dir [YOUR_DATA_DIR]
python val.py --name ${dataset}_UWTNET --output_dir

dataset=cvc
input_size=256
python train.py  --dataset ${dataset} --input_w ${input_size} --input_h ${input_size} --name ${dataset}_UWTNET  --data_dir [YOUR_DATA_DIR]
python val.py --name ${dataset}_UWTNET --output_dir

python val.py --name glas_UWTNet --output_dir ./VAL
python train.py  --dataset busi --input_w 256 --input_h 256 --name busi_UWTNET  --data_dir ./inputs




