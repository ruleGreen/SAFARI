for retriever_type in "dpr"
do
    CUDA_VISIBLE_DEVICES=2 python lora/infer.py \
        --retriever_type $retriever_type \
        --retriever_number 1
done