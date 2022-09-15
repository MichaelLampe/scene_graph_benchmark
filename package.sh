wget https://penzhanwu2.blob.core.windows.net/sgg/sgg_benchmark/vinvl_model_zoo/vinvl_vg_x152c4.pth
wget https://penzhanwu2.blob.core.windows.net/sgg/sgg_benchmark/vinvl_model_zoo/VG-SGG-dicts-vgoi6-clipped.json

CURRENT_GIT_COMMIT=`git rev-list --max-count=1 HEAD`
echo $CURRENT_GIT_COMMIT
tar -czvf scene_graph_benchmark-michael-lampe-fork-$CURRENT_GIT_COMMIT-with-weights.tgz scene_graph_benchmark/
