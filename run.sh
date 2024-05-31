export nproc_per_node=1
export nnodes=1
export node_rank=0

python -m torch.distributed.launch --nproc_per_node $nproc_per_node --nnodes $nnodes --node_rank $node_rank  ./main.py \
  --data-path path_to_BraTS_dataset
