BASE_DIRECTORY=scene_graph_benchmark
CURRENT_GIT_COMMIT=`git rev-list --max-count=1 HEAD`
FILE_NAME=scene_graph_benchmark-michael-lampe-fork-$CURRENT_GIT_COMMIT-with-weights.tgz

echo "Downloading relevant weights and label mappings"
wget -nc https://penzhanwu2.blob.core.windows.net/sgg/sgg_benchmark/vinvl_model_zoo/vinvl_vg_x152c4.pth
wget -nc https://penzhanwu2.blob.core.windows.net/sgg/sgg_benchmark/vinvl_model_zoo/VG-SGG-dicts-vgoi6-clipped.json

echo "Packaging commit $CURRENT_GIT_COMMIT"
cp ..
tar -czvf $FILE_NAME $BASE_DIRECTORY
cp FILE_NAME $BASE_DIRECTORY
cd $BASE_DIRECTORY
