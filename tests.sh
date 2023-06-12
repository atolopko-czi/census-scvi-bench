# params: obs_filter, gpu count, pytorch batch size, epochs
/usr/bin/time -v python census_scvi.py "tissue_general == 'lung' and is_primary_data == True" 1 128 1  > logs/lung_1gpu_128batch_1epoch.txt 2>&1
/usr/bin/time -v python census_scvi.py "tissue_general == 'lung' and is_primary_data == True" 2 128 1  > logs/lung_2gpu_128batch_1epoch.txt 2>&1
/usr/bin/time -v python census_scvi.py "tissue_general == 'lung' and is_primary_data == True" 4 128 1  > logs/lung_4gpu_128batch_1epoch.txt 2>&1
/usr/bin/time -v python census_scvi.py "tissue_general == 'lung' and is_primary_data == True" 1 256 1  > logs/lung_1gpu_256batch_1epoch.txt 2>&1
/usr/bin/time -v python census_scvi.py "tissue_general == 'lung' and is_primary_data == True" 1 512 1  > logs/lung_1gpu_512batch_1epoch.txt 2>&1
/usr/bin/time -v python census_scvi.py "tissue_general == 'lung' and is_primary_data == True" 4 512 1  > logs/lung_4gpu_512batch_1epoch.txt 2>&1
/usr/bin/time -v python census_scvi.py "tissue_general == 'lung' and is_primary_data == True" 4 512 10 > logs/lung_4gpu_512batch_10epoch.txt 2>&1
