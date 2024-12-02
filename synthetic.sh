my_list=("chair" "drums" "ficus" "hotdog" "lego" "materials" "mic" "ship")

for item in "${my_list[@]}"; do
    echo "Current test cases: $item"
    ./run.sh ~/dataset/nerf_synthetic/$item ./output/synthetic/$item
done