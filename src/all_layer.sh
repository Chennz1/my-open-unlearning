for i in $(seq 5 2 13)
do
  echo "Running layer index: $i"
  CUDA_VISIBLE_DEVICES=0 python steering_unlearn_v3_5_npo.py --layer_index $i 
done

# CUDA_VISIBLE_DEVICES=0 python steering_unlearn_v3_5_npo.py --layer_index 9
