#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=8:00:00
#SBATCH --account=p_scads_nlp
#SBATCH --job-name=masakhaner2_pipelines
#SBATCH --gres=gpu:1
#SBATCH --partition=alpha
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=end,fail

CFG_FILE=$1
lang=$2
SRC_NER_MODEL=$3
TGT_NER_MODEL=$4

source "$CFG_FILE"
export $(cut -d= -f1 "$CFG_FILE")
export NMTSCORE_CACHE=$WORKSPACE/cache

if [ -x "$(command -v module)" ]; then
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


if [ "$lang" = "swa" ]
then
    tgt_lang_code=swh_Latn
else
    tgt_lang_code=$lang"_Latn"
fi

# Cache translations and NER labelling for all pipeline's types
PIPE_CACHE_DIR=$WORKSPACE/cache/masakhaner2/$lang
mkdir -p $PIPE_CACHE_DIR

FWD_TRANS_PATH=$PIPE_CACHE_DIR/fwd_translation_nllb_${lang}_eng
if [ -f $FWD_TRANS_PATH ]; then
   echo "Use cached translation"
else
    echo "[PIPELINE] Start forward translation"
    python -m src.pipeline.run_pipeline experiment=partial/translate/nllb200_3B.yaml \
        src_lang=$lang tgt_lang=eng pipeline.translate.transform.src_lang_code=$tgt_lang_code \
        batch_size=$TRANS_BATCH_SIZE \
        dataset_path=masakhane/masakhaner2 \
        pipeline.load_ds.transform.split=test \
        pipeline.load_ds.transform.cfg_name=$lang \
        out_dir=$FWD_TRANS_PATH

    if [ $? -ne 0 ]; then
      echo "Error during forward translation!"
      exit 1
    fi
fi

ner_model_name=$(echo $SRC_NER_MODEL | sed  's\/\_\g')
SRC_ENTITIES_PATH=$PIPE_CACHE_DIR/src_entities_$ner_model_name
if [ -f $SRC_ENTITIES_PATH ]; then
   echo "Use cached source entities"
else
    echo "[PIPELINE] Start SRC NER labeling"
    python -m src.pipeline.run_pipeline log_to_wandb=false \
        pipeline=annotation/partial/src_ner \
        pipeline.load_translation.transform.dataset_path=$FWD_TRANS_PATH \
        pipeline.apply_ner.transform.model_path=$SRC_NER_MODEL \
        pipeline.apply_ner.transform.batch_size=$NER_ALIGN_BATCH_SIZE \
        pipeline.write_entities.transform.save_path=$SRC_ENTITIES_PATH

    if [ $? -ne 0 ]; then
      echo "Error during source NER labeling!"
      exit 1
    fi
fi


RUN="python -m src.pipeline.run_pipeline \
        log_to_wandb=true tgt_lang=$lang \
        pipeline.load_ds.transform.split=test \
        pipeline.load_ds.transform.dataset_path=masakhane/masakhaner2 \
        pipeline.load_ds.transform.cfg_name=$lang"

# Model transfer
echo "[PIPELINE] Start model transfer pipeline"
$RUN pipeline=annotation/full/model_transfer lang=$lang \
    pipeline.apply_ner.transform.model_path=$TGT_NER_MODEL \
    pipeline.apply_ner.transform.batch_size=$NER_ALIGN_BATCH_SIZE

# Compute maximum length of entities in the GT dataset
MAX_CAND_LENGTH=$(python3 $SRC_DIR/scripts/utils/count_max_entity_length.py -d masakhane/masakhaner2 -s test -c bam | awk 'FNR == 10 {print int($2)}')

RUN=$RUN pipeline.load_entities.transform.dataset_path=$SRC_ENTITIES_PATH \
    pipeline.cand_extraction.transform.max_words=$MAX_CAND_LENGTH

# NER score
$RUN pipeline=annotation/partial/ranges/ner \
    pipeline.cand_eval.transform.model_path=$TGT_NER_MODEL \
    pipeline.cand_eval.transform.batch_size=$NER_ALIGN_BATCH_SIZE

# NMT-score
$RUN pipeline=annotation/partial/ranges/nmtscore \
    pipeline.project.transform.cost_params[0].tgt_lang=$tgt_lang_code \
    pipeline.project.transform.cost_params[0].batch_size=$TRANS_BATCH_SIZE

# NER + NMT
$RUN pipeline=annotation/partial/ranges/nmtscore \
    pipeline.cand_eval.transform.model_path=$TGT_NER_MODEL \
    pipeline.cand_eval.transform.batch_size=$NER_ALIGN_BATCH_SIZE \
    pipeline.project.transform.cost_params[1].tgt_lang=$tgt_lang_code \
    pipeline.project.transform.cost_params[1].batch_size=$TRANS_BATCH_SIZE

aligners=(awesome_mbert)
for aligner in ${aligners[@]}
do
    ALIGNMENTS_PATH=$PIPE_CACHE_DIR/alignments_$aligner.arrow
    if [ -f $SRC_ENTITIES_PATH ]; then
       echo "Use cached $aligner alignments"
    else
        echo "[PIPELINE] Start W2W alignments computation"
        python -m src.pipeline.run_pipeline aligner=$aligner \
            log_to_wandb=false \
            pipeline=annotation/partial/alignments \
            pipeline.load_ds.transform.dataset_path=masakhane/masakhaner2 \
            pipeline.load_ds.transform.split=test \
            pipeline.load_ds.transform.cfg_name=$lang" \
            pipeline.load_entities.transform.dataset_path=$SRC_ENTITIES_PATH \
            pipeline.align_words.transform.batch_size=$NER_ALIGN_BATCH_SIZE \
            pipeline.save_alignments.transform.save_path=$ALIGNMENTS_PATH

        if [ $? -ne 0 ]; then
          echo "Error during w2w alignments calculation!"
          exit 1
        fi
    fi

    # Heuristic aligment-based
    $RUN aligner=$aligner pipeline=annotation/partial/heuristics \
        pipeline.load_alignments.transform.dataset_path=$ALIGNMENTS_PATH

    # Alignments-based score
    $RUN aligner=$aligner pipeline=annotation/partial/ranges/aligned_subranges \
        pipeline.load_alignments.transform.dataset_path=$ALIGNMENTS_PATH

    # NER + alignments (ignore NER labels)
    $RUN aligner=$aligner pipeline=annotation/partial/ranges/align_ner_fusion \
        pipeline.cand_eval.transform.model_path=$TGT_NER_MODEL \
        pipeline.cand_eval.transform.batch_size=$NER_ALIGN_BATCH_SIZE \
        pipeline.load_alignments.transform.dataset_path=$ALIGNMENTS_PATH

    # NER + alignments
    $RUN aligner=$aligner pipeline=annotation/partial/ranges/align_ner_per_class_fusion \
        pipeline.cand_eval.transform.model_path=$TGT_NER_MODEL \
        pipeline.cand_eval.transform.batch_size=$NER_ALIGN_BATCH_SIZE \
        pipeline.load_alignments.transform.dataset_path=$ALIGNMENTS_PATH

    # NMT + alignments
    $RUN aligner=$aligner pipeline=annotation/partial/ranges/align_nmtscore_fusion \
        pipeline.load_alignments.transform.dataset_path=$ALIGNMENTS_PATH \
        pipeline.project.transform.cost_params[1].tgt_lang=$tgt_lang_code \
        pipeline.project.transform.cost_params[1].batch_size=$TRANS_BATCH_SIZE

    # NMT + NER + alignments
    $RUN aligner=$aligner pipeline=annotation/partial/ranges/align_ner_nmtscore_fusion \
        pipeline.load_alignments.transform.dataset_path=$ALIGNMENTS_PATH \
        pipeline.cand_eval.transform.model_path=$TGT_NER_MODEL \
        pipeline.cand_eval.transform.batch_size=$NER_ALIGN_BATCH_SIZE \
        pipeline.project.transform.cost_params[2].tgt_lang=$tgt_lang_code \
        pipeline.project.transform.cost_params[2].batch_size=$TRANS_BATCH_SIZE

done

