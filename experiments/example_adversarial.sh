#!/usr/bin/env bash

# Runs an example of VGCN on CiteSeer in the adversarial case (loading the attacked graph from a file)
RESULTS_DIR='./results/test_adversarial' # Results for every epoch will be saved here
DATASET='citeseer_edgelist' # use _edgelist to load attacked graph, instead of original graph
EDGELIST='citeseer_attack_add_5000_v_6.gpickle' # file containing attacked graph

PYTHONPATH=. python experiments/run_vgcn.py --dataset-name=$DATASET  --initial-learning-rate=0.001 --dropout-rate=0.5 --l2-reg-scale-gcn=0.0005 --layer-type=dense --posterior-type=free --one-smoothing-factor=0.25 --edgelist=$EDGELIST --num-epochs=5000 --beta=0.01 --temperature-posterior=0.5 --temperature-prior=0.1 --use-half-val-to-train --log-every-n-iter=1  --results-dir=$RESULTS_DIR VGCN


# For polblogs we don't have a fixed split, hence use random-split
#DATASET='polblogs_edgelist' # use _edgelist to load attacked graph, instead of original graph
#EDGELIST='polblogs_attack_add_5000_v_6.gpickle' # file containing attacked graph
#PYTHONPATH=. python experiments/run_vgcn.py --dataset-name=$DATASET  --initial-learning-rate=0.001 --dropout-rate=0.5 --l2-reg-scale-gcn=0.0005 --layer-type=dense --posterior-type=free --one-smoothing-factor=0.25 --edgelist=$EDGELIST --num-epochs=5000 --beta=0.01 --temperature-posterior=0.5 --temperature-prior=0.1 --use-half-val-to-train --log-every-n-iter=1  --random-split --seed-np=85324 --results-dir=$RESULTS_DIR VGCN



