#!/bin/bash

declare -a C=("10" "100" "1000","3000")
declare -a n_trees=("5" "10" "100")
declare -a hs=("1" "2" "5","10")
declare -a max_depth=("1" "2")
declare -a layers=("1" "3" "5","10")

declare -a datasets=("Diabetes" "California" "Liver","KDD98")
declare -a models=("Boosted" "Cascade")
n_est = 50

for arg1 in "${C[@]}"; do
    for arg2 in "${n_trees[@]}"; do
        for arg3 in "${hs[@]}"; do
            for arg4 in "${max_depth[@]}"; do
                for arg5 in "${layers[@]}"; do
                    for arg6 in "${datasets[@]}"; do
                        for arg7 in "${models[@]}"; do
                            echo "Executing test arguments: $arg1 $arg2 $arg3 $arg4 $arg5 $arg6 $arg7"
                            python3 --C ${arg1} --n_trees ${arg2} --hs ${arg3} --max_depth ${arg3} --layers ${arg4} --datasets ${arg5} --models ${arg6}
                        done    
                    done    
                done
            done    
        done
    done    
done