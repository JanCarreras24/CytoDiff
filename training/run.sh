
GPU=1
#GPU="$1"
SET_SPLIT=1
SPLIT_IDX=15
#SPLIT_IDX="$2"

### ------------------
### Parameters
### ------------------
DATASET="matek" # eurosat
N_CLS=15
FEWSHOT_SEED="seed0"
N_SHOT=4
NUM_TRAIN_EPOCH=200


### ------------------
### Calculate CLASS_IDXS 
### ------------------
#START_RANGE=$(( (N_CLS / SET_SPLIT) * SPLIT_IDX)) # (15/5)*2 = 6
#END_RANGE=$(( (N_CLS / SET_SPLIT) * (SPLIT_IDX + 1) - 1 )) # ((15/5)*(2+1))-1) = 8
START_RANGE=1
END_RANGE=15

# Check if SPLIT_IDX is equal to SET_SPLIT - 1
if [ $SPLIT_IDX -eq $((SET_SPLIT - 1)) ]; then
    FINAL_END_RANGE=$((N_CLS - 1))
else
    FINAL_END_RANGE=$END_RANGE
fi

CLASS_IDXS=($(seq $START_RANGE $FINAL_END_RANGE))


### ------------------
### Run
### ------------------
for CLASS_IDX in "${CLASS_IDXS[@]}"; do

# for windows, use set CUDA_VISIBLE_DEVICES=$GPU, for linux, remove 'set' and line break
CUDA_VISIBLE_DEVICES=$GPU accelerate launch main.py \
--dataset=$DATASET \
--n_template=1 \
--fewshot_seed=$FEWSHOT_SEED \
--train_batch_size=8 \
--gradient_accumulation_steps=1 \
--learning_rate=1e-4 \
--lr_scheduler="cosine" \
--lr_warmup_steps=100 \
--num_train_epochs=$NUM_TRAIN_EPOCH \
--report_to="tensorboard" \
--train_text_encoder=True \
--is_tqdm=True \
--output_dir=output \
--n_shot=$N_SHOT \
--target_class_idx=$CLASS_IDX \
--resume_from_checkpoint=None \
$PARAM

done
