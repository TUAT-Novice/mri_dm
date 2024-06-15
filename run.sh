export nproc_per_node=4
export nnodes=1
export node_rank=0

data_path=path_to_brats_dataset
epoch=1000
batch_size=8
accm_freq=1
mri_ratio=2
model_path='./saved_models/'  # './saved_models/MODEL_NAME.h5' # to continue

python -m torch.distributed.launch --nproc_per_node $nproc_per_node --nnodes $nnodes --node_rank $node_rank  ./main.py \
  --data-path $data_path \
  -n $epoch \
  -b $batch_size \
  --accumulate-step $accm_freq \
  --mri_ratio $mri_ratio \
  --model_path $model_path

