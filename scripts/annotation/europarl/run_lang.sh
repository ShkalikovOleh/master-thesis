#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=8:00:00
#SBATCH --account=p_scads_nlp
#SBATCH --job-name=europarl_pipelines
#SBATCH --gres=gpu:1
#SBATCH --partition=alpha
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=end,fail

CFG_FILE=$1
lang=$2
TGT_NER_MODEL=$3

source "$CFG_FILE"
export $(cut -d= -f1 "$CFG_FILE")
export NMTSCORE_CACHE=$WORKSPACE/cache

if  [ -x "$(which sbatch)" ]; then
    # Execution on HPC
    module switch release/24.04
    module load GCCcore/13.2.0
    module load Python/3.11.5
    module load CUDA/12.1.1

    source $VENV_DIR/bin/activate

    nvidia-smi

    NER_ALIGN_BATCH_SIZE=256
    TRANS_BATCH_SIZE=32
else
    # Local execution
    NER_ALIGN_BATCH_SIZE=32
    TRANS_BATCH_SIZE=2
fi

PIPE_CACHE_DIR=$WORKSPACE/cache/europarl/$lang
mkdir -p $PIPE_CACHE_DIR

SRC_ENTITIES_PATH=$WORKSPACE/cache/europarl/en/src_entities/en

RUN="python -m src.pipeline.run_pipeline \
        log_to_wandb=true tgt_lang=$lang \
        pipeline.load_ds.transform.split=test \
        pipeline.load_ds.transform.dataset_path=ShkalikovOleh/europarl-ner \
        pipeline.load_ds.transform.cfg_name=$lang "

# Model transfer
echo "[PIPELINE] Start model transfer pipeline"
$RUN pipeline=annotation/full/model_transfer tgt_lang=$lang \
    pipeline.apply_ner.transform.model_path=$TGT_NER_MODEL \
    pipeline.apply_ner.transform.batch_size=$NER_ALIGN_BATCH_SIZE

# Compute maximum length of entities in the GT dataset
# MAX_CAND_LENGTH=$(python3 $SRC_DIR/scripts/utils/count_max_entity_length.py -d ShkalikovOleh/europarl-ner -s test -c $lang | awk 'FNR == 9 {print int($2)}')
MAX_CAND_LENGTH=10

aligners=(awesome_mbert)
for aligner in ${aligners[@]}
do
    ALIGNMENTS_PATH=$PIPE_CACHE_DIR/alignments_$aligner
    if [ -d $ALIGNMENTS_PATH ]; then
       echo "Use cached $aligner alignments"
    else
        echo "[PIPELINE] Start W2W alignments computation"
        python -m src.pipeline.run_pipeline aligner=$aligner \
            log_to_wandb=false \
            pipeline=annotation/partial/alignments \
            pipeline.load_ds.transform.dataset_path=ShkalikovOleh/europarl-ner \
            pipeline.load_ds.transform.split=test \
            pipeline.load_ds.transform.cfg_name=$lang \
            pipeline.load_entities.transform.dataset_path=$SRC_ENTITIES_PATH \
            pipeline.align_words.transform.batch_size=$NER_ALIGN_BATCH_SIZE \
            pipeline.save_alignments.transform.save_path=$ALIGNMENTS_PATH

        if [ $? -ne 0 ]; then
          echo "Error during w2w alignments calculation!"
          exit 1
        fi
    fi

    # Heuristic aligment-based
    python $SRC_DIR/scripts/annotation/europarl/run_heuristics.py \
        --langs $lang \
        --src-entities-path $SRC_ENTITIES_PATH\
        --align_path $ALIGNMENTS_PATH

    # ILP
    python $SRC_DIR/scripts/annotation/europarl/run_ilp.py \
        --lang $lang \
        --src-entities-path $SRC_ENTITIES_PATH \
        --align_path $ALIGNMENTS_PATH \
        --ner-model $TGT_NER_MODEL \
        --ner-batch-size $NER_ALIGN_BATCH_SIZE \
        --trans-batch-size $TRANS_BATCH_SIZE \
        --max-cand-length $MAX_CAND_LENGTH

done

