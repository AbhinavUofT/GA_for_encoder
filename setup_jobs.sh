#!/bin/bash
echo "Present working Directory is: `pwd`"

target_list=(unary random_unary GHZ Prime)
states=0
VAR="unary"
for qubits in 4 5 6 7 8 ; do
    echo "preparing file for $qubits qubits"
    mkdir ${qubits}_qubits
    for target in "${target_list[@]}"; do
        particles=1
        max_par=1
        if [ "$target" == "$VAR" ]; then
            echo $target
            max_par=$((qubits / 2))
        else
            max_par=1
        fi

        while [ $particles -le $max_par ] ; do
            mkdir ${qubits}_qubits/${target}_states_${particles}_particles
            cat main.py | sed "s|qqqq|$qubits|; s|kkkk|$target|; s|pppp|$particles|; s|ssss|$states|"  > ${qubits}_qubits/${target}_states_${particles}_particles/main.py
            cat job.sh | sed "s|kkkk|$target|" > ${qubits}_qubits/${target}_states_${particles}_particles/job.sh
            cp adapt_GA.py ${qubits}_qubits/${target}_states_${particles}_particles/
            cp GA.py ${qubits}_qubits/${target}_states_${particles}_particles/
            cp ccx_encoder.py ${qubits}_qubits/${target}_states_${particles}_particles/
            cp encoder_utils.py ${qubits}_qubits/${target}_states_${particles}_particles/
            cp evolve.py ${qubits}_qubits/${target}_states_${particles}_particles/
            cp fitness.py ${qubits}_qubits/${target}_states_${particles}_particles/
            cp GA_utils.py ${qubits}_qubits/${target}_states_${particles}_particles/
            cp general_utils.py ${qubits}_qubits/${target}_states_${particles}_particles/
            cp target_states.py ${qubits}_qubits/${target}_states_${particles}_particles/
            cd ${qubits}_qubits/${target}_states_${particles}_particles/
            sbatch job.sh
            cd ../../
            particles=$(($particles+1))
        done
    done
done
