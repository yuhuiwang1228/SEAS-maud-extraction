cache_type=split
eval_mode=test

for run_num in 1; do
  for epoch_num in 10; do
    for lr in 1e-4; do
      output_dir=./train_models/test_${cache_type}/roberta-base-maud-lr-$run_num
      model_dir=./train_models/test_${cache_type}/model-sub-$run_num-$epoch_num
      predict_dir=./train_models/test_${cache_type}/predict-sub-$run_num-$epoch_num
      train_file=./maud_data/maud_squad_${cache_type}_answers/maud_squad_train_sub_jason.json
      predict_file=./maud_data/maud_squad_${cache_type}_answers/maud_squad_test_sub_jason.json
      python train_new.py \
              --output_dir $output_dir \
              --model_dir $model_dir \
              --predict_dir $predict_dir \
              --model_type roberta \
              --model_name_or_path roberta-base \
              --train_file $train_file \
              --predict_file $predict_file \
              --cache_dir ./_cached_features/data-sub-$run_num \
              --version_2_with_negative \
              --learning_rate $lr \
              --num_train_epochs ${epoch_num} \
              --per_gpu_eval_batch_size=16  \
              --per_gpu_train_batch_size=32 \
              --max_seq_length 512 \
              --max_answer_length 512 \
              --doc_stride 256 \
              --save_steps 1000 \
              --threads 6 \
              --do_train \
              --do_eval \
              --do_freeze \
              --n_best_size 10 
      python evaluate.py -E test -T $predict_file $predict_dir
    done
  done
done

