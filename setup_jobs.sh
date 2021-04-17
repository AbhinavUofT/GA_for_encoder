#!/bin/bash
echo "Present working Directory is: `pwd`"

target_list=(unary random_unary)

for qubits in 4 6 8 10 12 ; do
    echo "preparing file for $qubits qubits"
    mkdir ${qubits}_qubits
    for target in "${target_list[@]}"; do
        mkdir ${qubits}_qubits/${target}_states
        cat GA.py | sed "s|xxxx|$qubits|; s|yyyy|$target|"  > ${qubits}_qubits/${target}_states/GA.py
        cp ccx_encoder.py ${qubits}_qubits/${target}_states/
        cp evolve.py ${qubits}_qubits/${target}_states/
        cp utils.py ${qubits}_qubits/${target}_states/
        cat job.sh | sed "s|xxxx|$qubits|; s|yyyy|$target|" > ${qubits}_qubits/${target}_states/job.sh
        cd ${qubits}_qubits/${target}_states
        sbatch job.sh
        cd ../../
    done
done
