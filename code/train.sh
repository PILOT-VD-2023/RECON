# *********************************************************************************
#RQ1:
python run_own.py --output_dir=./saved_models/RQ1_512 --model_type=roberta --tokenizer_name=microsoft/codebert-base --model_name_or_path=microsoft/codebert-base --do_train --train_data_file=../dataset/RQ1_512/train_cdata.jsonl --train_vul_data_file=../dataset/RQ1_512/train_vul_cdata.jsonl --eval_data_file=../dataset/RQ1_512/valid_all_cdata.jsonl --test_data_file=../dataset/RQ1_512/test_cdata.jsonl --epoch 12 --block_size 512 --train_batch_size 16 --eval_batch_size 64 --learning_rate 2e-5 --max_grad_norm 1.0 --evaluate_during_training --seed 123456 --cnn_size 128 --filter_size 3 --d_size 128 2>&1 | tee RQ1_512.log

python run_own.py --output_dir=./saved_models/RQ1_512 --model_type=roberta --tokenizer_name=microsoft/codebert-base --model_name_or_path=microsoft/codebert-base --do_test --train_data_file=../dataset/RQ1_512/train_cdata.jsonl --train_vul_data_file=../dataset/RQ1_512/train_vul_cdata.jsonl --eval_data_file=../dataset/RQ1_512/valid_all_cdata.jsonl --test_data_file=../dataset/RQ1_512/test_cdata.jsonl --epoch 12 --block_size 512 --train_batch_size 40 --eval_batch_size 64 --learning_rate 2e-5 --max_grad_norm 1.0 --evaluate_during_training --seed 123456 --cnn_size 128 --filter_size 3 --d_size 128 
 
python ../evaluator/evaluator.py -a ../dataset/RQ1_512/test_cdata.jsonl -p saved_models/RQ1_512/predictions.txt

# *********************************************************************************
#RQ3:
python run_own.py --output_dir=./saved_models/RQ1_512 --model_type=roberta --tokenizer_name=microsoft/codebert-base --model_name_or_path=microsoft/codebert-base --do_test --train_data_file=../dataset/RQ1_512/train_cdata.jsonl --train_vul_data_file=../dataset/RQ1_512/train_vul_cdata.jsonl --eval_data_file=../dataset/RQ1_512/valid_cdata.jsonl --test_data_file=../dataset/RQ1_512/test_vul_cdata.jsonl --epoch 8 --block_size 512 --train_batch_size 40 --eval_batch_size 64 --learning_rate 2e-5 --max_grad_norm 1.0 --evaluate_during_training --seed 123456 --cnn_size 128 --filter_size 3 --d_size 128 

python ../evaluator/evaluator.py -a ../dataset/RQ1_512/test_vul_cdata.jsonl -p saved_models/RQ1_512/predictions_vul.txt

python run_own.py --output_dir=./saved_models/RQ1_512 --model_type=roberta --tokenizer_name=microsoft/codebert-base --model_name_or_path=microsoft/codebert-base --do_test --train_data_file=../dataset/RQ1_512/train_cdata.jsonl --train_vul_data_file=../dataset/RQ1_512/train_vul_cdata.jsonl --eval_data_file=../dataset/RQ1_512/valid_cdata.jsonl --test_data_file=../dataset/RQ1_512/test_all_cdata.jsonl --epoch 16 --block_size 512 --train_batch_size 40 --eval_batch_size 64 --learning_rate 2e-5 --max_grad_norm 1.0 --evaluate_during_training --seed 123456 --cnn_size 128 --filter_size 3 --d_size 128 

python ../evaluator/evaluator.py -a ../dataset/RQ1_512/test_all_cdata.jsonl -p saved_models/RQ1_512/predictions_all.txt

# *********************************************************************************
#RQ2:

python run_own.py --output_dir=./saved_models/RQ2_512 --model_type=roberta --tokenizer_name=microsoft/codebert-base --model_name_or_path=microsoft/codebert-base --do_train --train_data_file=../dataset/RQ2_512/train_cdata.jsonl --train_vul_data_file=../dataset/RQ2_512/train_vul_cdata.jsonl --eval_data_file=../dataset/RQ2_512/valid_all_cdata.jsonl --test_data_file=../dataset/RQ2_512/test_cdata.jsonl --epoch 12 --block_size 512 --train_batch_size 16 --eval_batch_size 64 --learning_rate 2e-5 --max_grad_norm 1.0 --evaluate_during_training --seed 123456 --cnn_size 128 --filter_size 3 --d_size 128 2>&1 | tee RQ2_512.log

python run_own.py --output_dir=./saved_models/RQ2_512 --model_type=roberta --tokenizer_name=microsoft/codebert-base --model_name_or_path=microsoft/codebert-base --do_test --train_data_file=../dataset/RQ2_512/train_cdata.jsonl --train_vul_data_file=../dataset/RQ2_512/train_vul_cdata.jsonl --eval_data_file=../dataset/RQ2_512/valid_all_cdata.jsonl --test_data_file=../dataset/RQ2_512/test_cdata.jsonl --epoch 16 --block_size 512 --train_batch_size 40 --eval_batch_size 64 --learning_rate 2e-5 --max_grad_norm 1.0 --evaluate_during_training --seed 123456 --cnn_size 128 --filter_size 3 --d_size 128 

python ../evaluator/evaluator.py -a ../dataset/RQ2_512/test_cdata.jsonl -p saved_models/RQ2_512/predictions.txt

# *********************************************************************************

#RQ4:
python run.py --output_dir=./saved_models/RQ4_512 --model_type=roberta --tokenizer_name=microsoft/codebert-base --model_name_or_path=microsoft/codebert-base --do_train --train_data_file=../dataset/RQ4_512/train_cdata.jsonl --train_vul_data_file=../dataset/RQ4_512/train_vul_cdata.jsonl --eval_data_file=../dataset/RQ4_512/valid_all_cdata.jsonl --test_data_file=../dataset/RQ4_512/test_cdata.jsonl --epoch 12 --block_size 512 --train_batch_size 16 --eval_batch_size 64 --learning_rate 2e-5 --max_grad_norm 1.0 --evaluate_during_training --seed 123456 --cnn_size 128 --filter_size 3 --d_size 128 2>&1 | tee RQ4_512.log

python run.py --output_dir=./saved_models/RQ4_512 --model_type=roberta --tokenizer_name=microsoft/codebert-base --model_name_or_path=microsoft/codebert-base --do_test --train_data_file=../dataset/RQ4_512/train_cdata.jsonl --train_vul_data_file=../dataset/RQ4_512/train_vul_cdata.jsonl --eval_data_file=../dataset/RQ4_512/valid_cdata.jsonl --test_data_file=../dataset/RQ4_512/test_cdata.jsonl --epoch 12 --block_size 512 --train_batch_size 40 --eval_batch_size 64 --learning_rate 2e-5 --max_grad_norm 1.0 --evaluate_during_training --seed 123456 --cnn_size 128 --filter_size 3 --d_size 128 

python ../evaluator/evaluator.py -a ../dataset/RQ4_512/test_cdata.jsonl -p saved_models/RQ4_512/predictions.txt


