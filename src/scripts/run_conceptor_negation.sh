#!/bin/bash

model_version=bert-base-uncased
layer_index_list=12

# model_version=bert-large-uncased
# layer_index_list=24

# model_version=gpt2
# layer_index_list=12

# model_version=gpt2-large
# layer_index_list=36

# model_version=gpt-j
# layer_index_list=28

echo "Processing $model_version ..."
srun -n1 -N1 -c4 --exclusive --output=logs/cn_all_${model_version}.out python conceptor_negation.py \
    -task all \
    -model_version $model_version \
    -corpus_type_list brown sst reddit \
    -layer_index_list $layer_index_list \
    -wordlist_percentile_list 1 
