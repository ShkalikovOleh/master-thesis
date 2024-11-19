CFG_FILE=$1

if [[ $# -eq 1 ]]
then
    SRC_NER_MODEL=FacebookAI/xlm-roberta-large-finetuned-conll03-english
    TGT_NER_MODEL=$SRC_NER_MODEL
elif [[ $# -eq 2 ]]
then
    SRC_NER_MODEL=$2
    TGT_NER_MODEL=$SRC_NER_MODEL
elif [[ $# -eq 3 ]]
then
    SRC_NER_MODEL=$2
    TGT_NER_MODEL=$3
else
    echo "Illegal number of parameters" >&2
    exit 2
fi

source "$CFG_FILE"
export $(cut -d= -f1 "$CFG_FILE")

langs=(bam ewe fon hau ibo kin lug luo mos nya sna swa tsn twi wol xho yor zul)

for lang in ${langs[@]}
do
    sbatch $SRC_DIR/slurm/pipelines/masakhaner2/run_lang.sh $CFG_FILE $lang $SRC_NER_MODEL $TGT_NER_MODEL
done