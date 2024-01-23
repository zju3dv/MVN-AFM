MVN_AFM_DIR="/home/eric/Code/Private/AFM_Publish_Final/MVN_AFM_Final"  # The folder of MVN-AFM
DATA_NUM="9"      # the number of multi-view AFM images
GPU_NUM="0"       # the GPU index
DATA_NAME="MOF"


declare -a ITER_LIST=(
"0.3" "0.25" "0.2" "0.15" "0.1"
)

echo ${DATA_NAME}

cd ${MVN_AFM_DIR}
python init_data.py --input_folder load/${DATA_NAME}/

for THRESHOLD in ${ITER_LIST[@]}; do
echo ${THRESHOLD}

cd ${MVN_AFM_DIR}

python mv_align.py --input_folder load/${DATA_NAME}/ --data_num ${DATA_NUM}

python compute_mask.py --input_folder load/${DATA_NAME}/ --threshold ${THRESHOLD} --data_num ${DATA_NUM}

done

cd ${MVN_AFM_DIR}
python launch.py --config configs/afm.yaml --gpu ${GPU_NUM} --train dataset.scene=${DATA_NAME} tag=reconstruct trial_name=reconstruction name=${DATA_NAME} dataset.root_dir=./load/${DATA_NAME}/ 

