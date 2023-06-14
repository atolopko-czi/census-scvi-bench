egrep 'wall clock' logs/heart*txt > logs/selected_timings.txt
egrep 'run_training_epoch' logs/heart*txt >> logs/selected_timings.txt
egrep 'run_training_batch' logs/heart*txt >> logs/selected_timings.txt
egrep 'train_dataloader_next' logs/heart*txt >> logs/selected_timings.txt
egrep 'DDPStrategy.backward' logs/heart*txt >> logs/selected_timings.txt
egrep 'DDPStrategy.training_step' logs/heart*txt >> logs/selected_timings.txt
