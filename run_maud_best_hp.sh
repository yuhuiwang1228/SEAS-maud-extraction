cache_type=split
eval_mode=test

for run_num in 1; do
  for epoch_num in 4; do
    for lr in 1e-4; do
      output_dir=./train_models/test_${cache_type}/roberta-base-maud-lr-$lr
      predict_dir=./train_models/test_${cache_type}/predict/train_126_seg_uniq
      predict_file=./maud_data/maud_squad_${cache_type}_answers/maud_squad_train_126_seg_uniq.json
      python train.py \
              --output_dir $output_dir \
              --model_type roberta \
              --model_name_or_path roberta-base \
              --train_file ./maud_data/maud_squad_${cache_type}_answers/maud_squad_train_and_dev.json \
              --predict_file ./maud_data/maud_squad_${cache_type}_answers/maud_squad_test.json \
              --cache_dir ./_cached_features/train_126 \
              --version_2_with_negative \
              --learning_rate $lr \
              --num_train_epochs ${epoch_num} \
              --per_gpu_eval_batch_size=16  \
              --per_gpu_train_batch_size=40 \
              --max_seq_length 512 \
              --max_answer_length 512 \
              --doc_stride 256 \
              --save_steps 1000 \
              --overwrite_output_dir \
              --threads 6 \
              --do_train \
              --do_eval \
              --n_best_size 100
      python evaluate.py -E test -T $predict_file $predict_dir
    done
  done
done
