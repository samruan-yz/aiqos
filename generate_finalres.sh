res_max=2700
bert_max=192


for b in 10; do
    current_bert=$((bert_max * b / 100))

    for r in 90 85 80; do
        current_res=$((res_max * r / 100))

        #python3 CLITE.py $current_res $current_bert
        python3 PARTIES.py $current_res $current_bert 32 16
    done
done