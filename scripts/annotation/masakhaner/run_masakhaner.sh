CFG_FILE=$1

if [[ $# -eq 1 ]]
then
    SRC_NER_MODEL=ShkalikovOleh/mdeberta-v3-base-conll2003-en
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

MAX_CONCURRENT_SESSIONS=2 # gurobi WSL Academic license limits number of machines the solver can be run on simultaneously

langs=(bam ewe fon hau ibo kin lug luo mos nya sna swa tsn twi wol xho yor zul)

i=0
JOB_IDS=()
for lang in ${langs[@]}
do
    if [[ $i -ge $MAX_CONCURRENT_SESSIONS ]]
    then
        dep_idx=$(($i - $MAX_CONCURRENT_SESSIONS))
        JOBID=${JOB_IDS[$dep_idx]}
        deps="-d afterany:$JOBID"
    else
        deps=""
    fi

    sbatch_out=$(sbatch $deps $SRC_DIR/scripts/annotation/masakhaner/run_lang.sh $CFG_FILE $lang $SRC_NER_MODEL $TGT_NER_MODEL)
    JOBID=$(sbatch_out | awk '{print $4}')
    JOB_IDS+=($JOBID)

    i=$((i+1))
done