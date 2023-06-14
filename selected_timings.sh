egrep 'wall clock' logs/heart_?gpu_128batch_1epoch*txt > logs/selected_timings.txt
echo >> logs/selected_timings.txt
egrep 'Retrieved batch' logs/heart_?gpu_128batch_1epoch*txt >> logs/selected_timings.txt
echo >> logs/selected_timings.txt
egrep 'Total' logs/heart_?gpu_128batch_1epoch*txt >> logs/selected_timings.txt
echo >> logs/selected_timings.txt
egrep 'run_training_epoch' logs/heart_?gpu_128batch_1epoch*txt >> logs/selected_timings.txt
egrep 'run_training_batch' logs/heart_?gpu_128batch_1epoch*txt >> logs/selected_timings.txt
egrep 'train_dataloader_next' logs/heart_?gpu_128batch_1epoch*txt >> logs/selected_timings.txt
egrep 'DDPStrategy.backward' logs/heart_?gpu_128batch_1epoch*txt >> logs/selected_timings.txt
egrep 'DDPStrategy.training_step' logs/heart_?gpu_128batch_1epoch*txt >> logs/selected_timings.txt
