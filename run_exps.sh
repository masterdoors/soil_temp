declare -a C=("10000")
#declare -a n_trees=("5" "10" "100")
declare -a n_trees=("4")
#declare -a hs=("1" "2" "5" "10")
declare -a hs=("3")
#declare -a max_depth=("1" "2")
declare -a max_depth=("1")
#declare -a layers=("1" "3" "5" "10")
declare -a layers=("100")

#declare -a datasets=("Diabetes" "California" "Liver" "KDD98")
declare -a datasets=("KDD98")
declare -a models=("BOOSTED")
declare  n_est=50

declare -a C0=("0")
declare -a hs0=("0")
declare -a n_trees0=("0")
declare -a models0=("Cascade")
declare -a models1=("XGB")

#declare -a layers0=("1" "3" "5" "10" "50" "100")
declare -a layers0=("10")

for arg1 in "${C[@]}"; do
    for arg2 in "${n_trees[@]}"; do
        for arg3 in "${hs[@]}"; do
            for arg4 in "${max_depth[@]}"; do
                for arg5 in "${layers[@]}"; do
                    for arg6 in "${datasets[@]}"; do
                        for arg7 in "${models[@]}"; do
                            echo "Executing test arguments: $arg1 $arg2 $arg3 $arg4 $arg5 $arg6 $arg7"
                            python3 standard_datasets.py --C ${arg1} --n_trees ${arg2} --hs ${arg3} --max_depth ${arg4} --layers ${arg5} --dataset ${arg6} --model ${arg7}
                        done    
                    done    
                done
            done    
        done
    done    
done

# +
# for arg1 in "${C0[@]}"; do
#     for arg2 in "${n_trees[@]}"; do
#         for arg3 in "${hs0[@]}"; do
#             for arg4 in "${max_depth[@]}"; do
#                 for arg5 in "${layers[@]}"; do
#                     for arg6 in "${datasets[@]}"; do
#                         for arg7 in "${models0[@]}"; do
#                             echo "Executing test arguments: $arg1 $arg2 $arg3 $arg4 $arg5 $arg6 $arg7"
#                             python3 standard_datasets.py --C ${arg1} --n_trees ${arg2} --hs ${arg3} --max_depth ${arg4} --layers ${arg5} --dataset ${arg6} --model ${arg7}
#                         done    
#                     done    
#                 done
#             done    
#         done
#     done    
# done

# +
# for arg1 in "${C0[@]}"; do
#     for arg2 in "${n_trees0[@]}"; do
#         for arg3 in "${hs0[@]}"; do
#             for arg4 in "${max_depth[@]}"; do
#                 for arg5 in "${layers0[@]}"; do
#                     for arg6 in "${datasets[@]}"; do
#                         for arg7 in "${models1[@]}"; do
#                             echo "Executing test arguments: $arg1 $arg2 $arg3 $arg4 $arg5 $arg6 $arg7"
#                             python3 standard_datasets.py --C ${arg1} --n_trees ${arg2} --hs ${arg3} --max_depth ${arg4} --layers ${arg5} --dataset ${arg6} --model ${arg7}
#                         done    
#                     done    
#                 done
#             done    
#         done
#     done    
# done
