#!/bin/bash

source /ext3/env.sh
conda activate diffdock-pocket
jeg_dir=/scratch/projects/yzlab/group/kinase_eric/kinase_location_classifier/jeg10045_kinase_scripts/diffdock/DiffDock/

cd /scratch/eac709/overlays/jeg/DiffDock-Confidence-Test

# 250416_diffdock-tune
# python gen_cache.py --config default_inference_args.yaml --protein_file protein \
#     --pdbbind_esm_embeddings_path /vast/eac709/allo_crossdock/DiffDock-L/AlloSet/241120/data/esm2/ \
#     --pdbbind_dir $jeg_dir/data/allosteric_dataset/ \
#     --split_train $jeg_dir/allo_train_split.txt --split_val $jeg_dir/allo_test_val_split.txt \
#     --cache_path data/alloset_only_cache_tor_weights/ \
#     --n_epochs 1 --log_dir workdir/delete_me --num_dataloader_workers 12 --dataloader_prefetch_factor 100

split_dir=/scratch/projects/yzlab/group/kinase_eric/kinase_location_classifier/final_allo_only_splits_250604/

# python gen_cache.py --config default_inference_args.yaml --protein_file protein \
#     --pdbbind_esm_embeddings_path /vast/eac709/allo_crossdock/DiffDock-L/AlloSet/241120/data/esm2/ \
#     --pdbbind_dir $jeg_dir/data/allosteric_dataset/ \
#     --split_train $split_dir/train.txt \
#     --split_val $split_dir/val.txt \
#     --cache_path data/final_allo_only_splits_250604_trainval/ \
#     --n_epochs 1 --log_dir workdir/delete_me --num_dataloader_workers 12 --dataloader_prefetch_factor 100

# python gen_cache.py --config default_inference_args.yaml --protein_file protein \
#     --pdbbind_esm_embeddings_path /vast/eac709/allo_crossdock/DiffDock-L/AlloSet/241120/data/esm2/ \
#     --pdbbind_dir $jeg_dir/data/allosteric_dataset/ \
#     --split_train weird_eval/test_split.txt \
#     --split_val weird_eval/test_split.txt \
#     --cache_path weird_eval/cache/ \
#     --n_epochs 1 --log_dir workdir/delete_me --num_dataloader_workers 12 --dataloader_prefetch_factor 100

# python gen_cache.py --config default_inference_args.yaml --protein_file protein \
#     --pdbbind_esm_embeddings_path /vast/eac709/allo_crossdock/DiffDock-L/AlloSet/241120/data/esm2/ \
#     --pdbbind_dir $jeg_dir/data/allosteric_dataset/ \
#     --split_train $split_dir/train.txt \
#     --split_val $split_dir/val.txt \
#     --cache_path data/final_allo_only_splits_250612_trainval/ \
#     --n_epochs 1 --log_dir workdir/delete_me --num_dataloader_workers 12 --dataloader_prefetch_factor 100

# python gen_cache.py --config default_inference_args.yaml --protein_file protein \
#     --pdbbind_esm_embeddings_path /vast/eac709/allo_crossdock/DiffDock-L/AlloSet/241120/data/esm2/ \
#     --pdbbind_dir $jeg_dir/data/allosteric_dataset/ \
#     --split_train $split_dir/test_all_kinase_binders.txt \
#     --split_val $split_dir/test_all_kinase_binders.txt \
#     --cache_path data/final_allo_only_splits_250612_test/ \
#     --n_epochs 1 --log_dir workdir/delete_me --num_dataloader_workers 12 --dataloader_prefetch_factor 100

# python gen_cache.py --config default_inference_args.yaml --protein_file protein \
#     --pdbbind_esm_embeddings_path /vast/eac709/allo_crossdock/DiffDock-L/AlloSet/250612/data/250610_corrected_sdf_supersample/esm2/ \
#     --pdbbind_dir /vast/eac709/allo_crossdock/250610_corrected_sdf_supersample/ \
#     --split_train $split_dir/train_supersample.txt \
#     --split_val $split_dir/val.txt \
#     --cache_path data/final_allo_only_splits_250612_supersample/ \
#     --n_epochs 1 --log_dir workdir/delete_me --num_dataloader_workers 12 --dataloader_prefetch_factor 100

python gen_cache.py --config default_inference_args.yaml --protein_file protein \
    --pdbbind_esm_embeddings_path /vast/eac709/allo_crossdock/DiffDock-L/AlloSet/250612/data/250610_corrected_sdf_supersample/esm2/ \
    --pdbbind_dir /vast/eac709/allo_crossdock/250610_corrected_sdf_supersample/ \
    --split_train $split_dir/train_supersample_x2.txt \
    --split_val $split_dir/val.txt \
    --cache_path data/final_allo_only_splits_250612_supersample_x2/ \
    --n_epochs 1 --log_dir workdir/delete_me --num_dataloader_workers 12 --dataloader_prefetch_factor 100

# run this after generating the proper input pdbbind_dir
# python gen_cache.py --config default_inference_args.yaml --protein_file protein \
#     --pdbbind_esm_embeddings_path /vast/eac709/allo_crossdock/DiffDock-L/AlloSet/250612/data/250610_corrected_sdf_supersample/esm2/ \
#     --pdbbind_dir /vast/eac709/allo_crossdock/250610_corrected_sdf_supersample/ \
#     --split_train $split_dir/train_supersample_x4.txt \
#     --split_val $split_dir/val.txt \
#     --cache_path data/final_allo_only_splits_250612_supersample_x4/ \
#     --n_epochs 1 --log_dir workdir/delete_me --num_dataloader_workers 12 --dataloader_prefetch_factor 100

# python gen_cache.py --config default_inference_args.yaml --protein_file protein \
#     --pdbbind_esm_embeddings_path /vast/eac709/allo_crossdock/DiffDock-L/AlloSet/250612/data/250610_corrected_sdf_supersample/esm2/ \
#     --pdbbind_dir /vast/eac709/allo_crossdock/250610_corrected_sdf_supersample/ \
#     --split_train $split_dir/train_supersample_x8.txt \
#     --split_val $split_dir/val.txt \
#     --cache_path data/final_allo_only_splits_250612_supersample_x8/ \
#     --n_epochs 1 --log_dir workdir/delete_me --num_dataloader_workers 12 --dataloader_prefetch_factor 100



### 
# python gen_cache.py --config default_inference_args.yaml --protein_file protein \
#     --pdbbind_esm_embeddings_path /vast/eac709/allo_crossdock/DiffDock-L/AlloSet/241120/data/esm2/ \
#     --pdbbind_dir $jeg_dir/data/allosteric_dataset/ \
#     --split_train $split_dir/train.txt \
#     --split_val $split_dir/test_all_kinase_binders.txt \
#     --cache_path /vast/eac709/allo_crossdock/DiffDock-L/AlloSet/250605/cache/test_all_kinase_binders/ \
#     --n_epochs 1 --log_dir workdir/delete_me --num_dataloader_workers 12 --dataloader_prefetch_factor 100

# python gen_cache.py --config default_inference_args.yaml --protein_file protein --pdbbind_esm_embeddings_path /vast/eac709/allo_crossdock/DiffDock-L/AlloSet/241120/data/esm2/ \
#     --pdbbind_dir $jeg_dir/data/allosteric_dataset/ --split_train 250325_hyperparam-alloset/type1_train.txt --split_val 250325_hyperparam-alloset/type1_val.txt \
#     --cache_path 250325_hyperparam-alloset/cache_conf_model \
#     --n_epochs 1 --log_dir workdir/delete_me --num_dataloader_workers 12 --dataloader_prefetch_factor 100 \
#     --all_atoms --atom_radius 5 --atom_max_neighbors 8 --max_radius 5 --c_alpha_max_neighbors 24

# python gen_cache.py --config default_inference_args.yaml --protein_file protein --pdbbind_esm_embeddings_path /vast/eac709/allo_crossdock/DiffDock-L/AlloSet/241120/data/esm2/ \
#     --pdbbind_dir $jeg_dir/data/allosteric_dataset/ --split_train 250415_hyperp-allo-alloset/type3_train.txt --split_val 250415_hyperp-allo-alloset/type3_val.txt \
#     --cache_path 250415_hyperp-allo-alloset/cache \
#     --n_epochs 1 --log_dir workdir/delete_me --num_dataloader_workers 12 --dataloader_prefetch_factor 100

# python gen_cache.py --config default_inference_args.yaml --protein_file protein \
#     --pdbbind_esm_embeddings_path /vast/eac709/allo_crossdock/DiffDock-L/AlloSet/241120/data/esm2/ \
#     --pdbbind_dir $jeg_dir/data/allosteric_dataset/ --split_train 250415_hyperp-allo-alloset/type3_train.txt --split_val 250415_hyperp-allo-alloset/type3_test.txt \
#     --cache_path 250415_hyperp-allo-alloset/cache_test \
#     --n_epochs 1 --log_dir workdir/delete_me --num_dataloader_workers 12 --dataloader_prefetch_factor 100

##############

# python gen_cache.py --config default_inference_args.yaml --protein_file protein --pdbbind_esm_embeddings_path /vast/eac709/allo_crossdock/DiffDock-L/AlloSet/241120/data/esm2/ \
#     --pdbbind_dir $jeg_dir/data/allosteric_dataset/ --split_train 250325_hyperparam-alloset/type1_train.txt --split_val 250325_hyperparam-alloset/type1_val.txt \
#     --cache_path 250325_hyperparam-alloset/cache \
#     --n_epochs 1 --log_dir workdir/delete_me --num_dataloader_workers 12 --dataloader_prefetch_factor 100

# python gen_cache.py --config default_inference_args.yaml --protein_file protein --pdbbind_esm_embeddings_path /vast/eac709/allo_crossdock/DiffDock-L/AlloSet/241120/data/esm2/ \
#     --pdbbind_dir $jeg_dir/data/allosteric_dataset/ --split_train 250325_hyperparam-alloset/type1_train.txt --split_val 250325_hyperparam-alloset/type1_test.txt \
#     --cache_path 250325_hyperparam-alloset/cache_test \
#     --n_epochs 1 --log_dir workdir/delete_me --num_dataloader_workers 12 --dataloader_prefetch_factor 100


# python gen_cache.py --config default_inference_args.yaml --protein_file protein --pdbbind_esm_embeddings_path /vast/eac709/allo_crossdock/DiffDock-L/AlloSet/241120/data/esm2/ \
#     --pdbbind_dir $jeg_dir/data/allosteric_dataset/ --split_train 250415_hyperp-allo-alloset/type3_train.txt --split_val 250415_hyperp-allo-alloset/type3_val.txt \
#     --cache_path 250415_hyperp-allo-alloset/cache_conf_model \
#     --n_epochs 1 --log_dir workdir/delete_me --num_dataloader_workers 12 --dataloader_prefetch_factor 100 \
#     --all_atoms --atom_radius 5 --atom_max_neighbors 8 --max_radius 5 --c_alpha_max_neighbors 24

# python gen_cache.py --config default_inference_args.yaml --protein_file protein --pdbbind_esm_embeddings_path /vast/eac709/allo_crossdock/DiffDock-L/AlloSet/241120/data/esm2/ \
#     --pdbbind_dir $jeg_dir/data/allosteric_dataset/ --split_train 250416_diffdock-tune/allo_train_split.txt --split_val 250416_diffdock-tune/allo_val_split.txt \
#     --cache_path 250419_s3w0iwik_eval/eval_cache \
#     --n_epochs 1 --log_dir workdir/delete_me --num_dataloader_workers 12 --dataloader_prefetch_factor 100

# python gen_cache.py --config default_inference_args.yaml --protein_file protein --pdbbind_esm_embeddings_path /vast/eac709/allo_crossdock/DiffDock-L/AlloSet/241120/data/esm2/ \
#     --pdbbind_dir $jeg_dir/data/allosteric_dataset/ --split_train 250416_diffdock-tune/type3_train.txt --split_val 250416_diffdock-tune/type3_test.txt \
#     --cache_path 250416_diffdock-tune/cache_test \
#     --n_epochs 1 --log_dir workdir/delete_me --num_dataloader_workers 12 --dataloader_prefetch_factor 100

# python gen_cache.py --config default_inference_args.yaml --protein_file protein --pdbbind_esm_embeddings_path /vast/eac709/allo_crossdock/DiffDock-L/AlloSet/241120/data/esm2/ \
#     --pdbbind_dir $jeg_dir/data/allosteric_dataset/ --split_train 250416_diffdock-tune/allo_train_split.txt --split_val 250416_diffdock-tune/allo_val_split.txt \
#     --cache_path 250419_s3w0iwik_eval/eval_cache_conf_model \
#     --n_epochs 1 --log_dir workdir/delete_me --num_dataloader_workers 12 --dataloader_prefetch_factor 100 \
#     --all_atoms --atom_radius 5 --atom_max_neighbors 8 --max_radius 5 --c_alpha_max_neighbors 24

#python gen_cache.py --config default_inference_args.yaml --protein_file protein --pdbbind_esm_embeddings_path data/esm_embedding_output --n_epochs 1 --log_dir workdir/delete_me --num_dataloader_workers 48 --dataloader_prefetch_factor 100
#python gen_cache.py --config default_inference_args.yaml --protein_file protein --pdbbind_esm_embeddings_path /scratch/projects/yzlab/group/kinase_eric/kinase_location_classifier/kinase_md/supersampled_complexes_esm/embeddings/ --pdbbind_dir /scratch/projects/yzlab/group/kinase_eric/kinase_location_classifier/kinase_md/supersampled_complexes --split_train /scratch/projects/yzlab/group/kinase_eric/kinase_location_classifier/kinase_md/supersampled_complexes/complex_names.txt --split_val allo_test_split.txt --n_epochs 1 --log_dir workdir/delete_me --num_dataloader_workers 48 --dataloader_prefetch_factor 100
#python gen_cache.py --config default_inference_args.yaml --protein_file protein --pdbbind_esm_embeddings_path /vast/eac709/allo_crossdock/DiffDock-L/AlloSet/241120/data/esm2/ --pdbbind_dir /scratch/projects/yzlab/group/kinase_eric/kinase_location_classifier/kinase_md/supersampled_complexes --split_train /scratch/projects/yzlab/group/kinase_eric/kinase_location_classifier/kinase_md/supersampled_complexes/complex_names.txt --split_val allo_test_split.txt --n_epochs 1 --log_dir workdir/delete_me --num_dataloader_workers 1 --dataloader_prefetch_factor 100
#python gen_cache.py --config default_inference_args.yaml --protein_file protein --pdbbind_esm_embeddings_path /scratch/projects/yzlab/group/kinase_eric/kinase_location_classifier/kinase_md/super_snp_esm/embeddings/ --pdbbind_dir /scratch/projects/yzlab/group/kinase_eric/kinase_location_classifier/kinase_md/super_snp --split_train /scratch/projects/yzlab/group/kinase_eric/kinase_location_classifier/kinase_md/super_snp/complex_names.txt --split_val allo_test_split.txt --n_epochs 1 --log_dir workdir/delete_me --num_dataloader_workers 48 --dataloader_prefetch_factor 100
#python gen_cache.py --config default_inference_args.yaml --protein_file protein --pdbbind_esm_embeddings_path /scratch/projects/yzlab/group/kinase_eric/kinase_location_classifier/kinase_md/super_snp_esm/embeddings/ --pdbbind_dir /scratch/projects/yzlab/group/kinase_eric/kinase_location_classifier/kinase_md/supersampled_complexes --split_train /scratch/projects/yzlab/group/kinase_eric/kinase_location_classifier/kinase_md/supersampled_complexes/complex_names.txt --split_val allo_test_split.txt --n_epochs 1 --log_dir workdir/delete_me --num_dataloader_workers 48 --dataloader_prefetch_factor 100
#python gen_cache.py --config default_inference_args.yaml --protein_file protein --pdbbind_esm_embeddings_path /scratch/projects/yzlab/group/kinase_eric/kinase_location_classifier/decoy_experiment/train_data_esm/embeddings/ --pdbbind_dir /scratch/projects/yzlab/group/kinase_eric/kinase_location_classifier/decoy_experiment/train_data --split_train /scratch/projects/yzlab/group/kinase_eric/kinase_location_classifier/decoy_experiment/train_data/complex_names.txt --split_val allo_test_split.txt --n_epochs 1 --log_dir workdir/delete_me --num_dataloader_workers 48 --dataloader_prefetch_factor 100
#python gen_cache.py --config default_inference_args.yaml --protein_file protein --pdbbind_esm_embeddings_path /scratch/projects/yzlab/group/kinase_eric/kinase_location_classifier/kinase_md/debug_esm/embeddings/ --pdbbind_dir /scratch/projects/yzlab/group/kinase_eric/kinase_location_classifier/kinase_md/debug --split_train /scratch/projects/yzlab/group/kinase_eric/kinase_location_classifier/kinase_md/debug/complex_names.txt --split_val allo_test_split.txt --n_epochs 1 --log_dir workdir/delete_me --num_dataloader_workers 48 --dataloader_prefetch_factor 100
#python gen_cache.py --config default_inference_args.yaml --protein_file protein --pdbbind_esm_embeddings_path /scratch/projects/yzlab/group/kinase_eric/kinase_location_classifier/kinase_md/combo_esm/embeddings/ --pdbbind_dir /scratch/projects/yzlab/group/kinase_eric/kinase_location_classifier/kinase_md/combo --split_train /scratch/projects/yzlab/group/kinase_eric/kinase_location_classifier/kinase_md/combo/complex_names.txt --split_val allo_test_split.txt --n_epochs 1 --log_dir workdir/delete_me --num_dataloader_workers 48 --dataloader_prefetch_factor 100
# python gen_cache.py --config default_inference_args.yaml --protein_file protein --pdbbind_esm_embeddings_path /scratch/projects/yzlab/group/kinase_eric/kinase_location_classifier/kinase_md/boost_esm/embeddings/ --pdbbind_dir /scratch/projects/yzlab/group/kinase_eric/kinase_location_classifier/kinase_md/boost --split_train /scratch/projects/yzlab/group/kinase_eric/kinase_location_classifier/kinase_md/boost/complex_names.txt --split_val allo_test_split.txt --n_epochs 1 --log_dir workdir/delete_me --num_dataloader_workers 48 --dataloader_prefetch_factor 100


rm -rf workdir/delete_me
