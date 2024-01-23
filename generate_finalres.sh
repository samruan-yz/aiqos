res_max=52
bert_max=10


for b in 10; do
    current_bert=$((bert_max * b / 100))

    for r in 75 70 65; do
        current_res=$((res_max * r / 100))

        #python3 CLITE.py $current_res $current_bert
        python3 PARTIES.py $current_res $current_bert 
    done
done