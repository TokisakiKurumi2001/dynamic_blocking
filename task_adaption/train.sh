CUDA_VISIBLE_DEVICES=0 \
python train.py \
    --model_name_or_path VietAI/vit5-base \
    --do_train \
    --do_eval \
    --do_predict \
    --source_lang corrupted \
    --target_lang full \
    --train_file data/train.json \
    --validation_file data/valid.json \
    --test_file data/test.json \
    --max_seq_length 256 \
    --output_dir ~/dynamic_blocking/task_adaption_model/vit5/v1 \
    --per_device_train_batch_size=8 \
    --per_device_eval_batch_size=8 \
    --overwrite_output_dir \
    --predict_with_generate \
    --save_total_limit 3 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end True \
    --metric_for_best_model eval_loss \
    --num_train_epochs 10
