#!/bin/bash -l

################# Part-1 Slurm directives ####################
## Working dir
#SBATCH -D "/users/js3006/conddiff/UD2M"
## Environment variables
#SBATCH --export=ALL
## Output and Error Files
##SBATCH -o job-%j.output
#SBATCH -o o_train_dif.output
#SBATCH -e e_train_dif.error
## Job name
#SBATCH -J CDM_SR4
## Run time: "hours:minutes:seconds", "days-hours"
#SBATCH --time=72:00:00
## Memory limit (in megabytes). Total --mem or amount per cpu --mem-per-cpu
#SBATCH --mem=16000
## GPU requirements
#SBATCH --gres gpu:1
##SBATCH --ntasks=8
## Specify partition
#SBATCH -p gpu

################# Part-2 Shell script ####################
#===============================
#  Activate Flight Environment
#-------------------------------
source "${flight_ROOT:-/opt/flight}"/etc/setup.sh

#==============================
#  Activate Package Ecosystem
#------------------------------
# Load the Conda module for access to `conda` command
# flight env activate conda

# Activate the specific Conda environment
flight env activate gridware
module load apps/anaconda3
module load apps/nvidia-cuda/11.2.2
source activate base
conda activate Cond_diff

if [ $? -eq 0 ]; then
    echo "Conda is active and functioning."
else
    echo "Conda did not initialize properly."
    exit 1
fi

###parameters
dataset_name="MNIST"
task="sr"
sigma_model=0.05

#===========================
#  Create results directory
#---------------------------
RESULTS_DIR1="/users/js3006/conddiff/UD2M/${SLURM_JOB_ID}"
RESULTS_DIR=$RESULTS_DIR1"/${SLURM_JOB_ID}"
echo "Your results will be stored in: $RESULTS_DIR"
mkdir -p "$RESULTS_DIR1"

#===============================
#  Application launch commands
#-------------------------------
# python3 main.py \
#                 --task $task \
#                 --operator_name "gaussian" \
#                 --lambda_ 0.1 \
#                 --config_file "temp_$i.yaml" \
#                 --dataset_name $dataset_name \
#                 --sigma_model $sigma_model \
#                 --max_unfolded_iter $i \
#                 --kernel_size 3 \
#                 --learned_lambda_sig True \
#                 --sigma_model $sigma_model \
#             > "$RESULTS_DIR"_out_train_dif.output \
#             2> "$RESULTS_DIR"_err_train_dif.error


# for num_timesteps in {1,} ; do
#     learned_lambda_sig=True
#     general=False

#     ckpt_dir="/users/js3006/conddiff/UD2M/Results/ckpts_ud2m_sr"
#     ckpt_name="mnist_lora_sr_gaussian_epoch=999-5.ckpt"
#     op="gaussian"            # gaussian, uniform, motion



#     config_file="fine_tuning_lmodel.yaml"



#     # python3 sampling.py \
#     #                 --task $task \
#     #                 --operator_name $op \
#     #                 --lambda_ 0.1 \
#     #                 --sigma_model $sigma_model \
#     #                 --config_file $config_file \
#     #                 --dataset_name $dataset_name \
#     #                 --ckpt_dir "$ckpt_dir" \
#     #                 --ckpt_name "$ckpt_name" \
#     #                 --num_timesteps $num_timesteps \
#     #                 --max_images 10000 \
#     #                 --kernel_size 3 \
#     #                 --max_unfolded_iter 3 \
#     #                 --learned_lambda_sig $learned_lambda_sig \
#     #                 --num_samples 4 \
#     #     > "$RESULTS_DIR"_out_sample_dif.output \
#     #     2> "$RESULTS_DIR"_err_sample_dif.error
        

#     echo "Computing FID: lambda: $lamb and sigma $sigma on dataset_name $dataset_name"          
#     python3 fid.py \
#             -d Results/MNIST/sr_operator_gaussian/ddim/K_3_N_$num_timesteps \
#         > "$RESULTS_DIR"_out_fid.output \
#         2> "$RESULTS_DIR"_err_fid.error


# done
for i in {2,}; do
sed 's|users/js3006/conddiff/UD2M/Results|users/js3006/conddiff/UD2M/Results/ablation_'"${i}"'|g' configs/fine_tuning_lmodel.yaml > configs/temp_$i.yaml
# python3 main.py \
#                 --task $task \
#                 --operator_name "gaussian" \
#                 --lambda_ 0.1 \
#                 --config_file "temp_$i.yaml" \
#                 --dataset_name $dataset_name \
#                 --sigma_model $sigma_model \
#                 --max_unfolded_iter $i \
#                 --kernel_size 3 \
#                 --learned_lambda_sig True \
#                 --sigma_model $sigma_model \
#             > "$RESULTS_DIR"_out_train_dif_$i.output \
#             2> "$RESULTS_DIR"_err_train_dif_$i.error


    learned_lambda_sig=True
    general=False

    ckpt_dir=/users/js3006/conddiff/UD2M/Results/ablation_${i}/ckpts_ud2m_sr
    ckpt_name="mnist_lora_sr_gaussian_epoch=399-5.ckpt"
    op="gaussian"            # gaussian, uniform, motion

    # Use 16/i timesteps  for a total of 16 NFEs
    for num_timesteps in {$((16/i)),}; do
        python3 sampling.py \
                        --task $task \
                        --operator_name $op \
                        --lambda_ 0.1 \
                        --sigma_model $sigma_model \
                        --config_file "temp_$i.yaml" \
                        --dataset_name $dataset_name \
                        --ckpt_dir "$ckpt_dir" \
                        --ckpt_name "$ckpt_name" \
                        --num_timesteps $num_timesteps \
                        --max_images 10000 \
                        --kernel_size 3 \
                        --max_unfolded_iter $i \
                        --learned_lambda_sig $learned_lambda_sig \
                        --num_samples 512 \
                        --equivariant\
            > "$RESULTS_DIR"_out_sample_dif_samp_K_${i}_N_${num_timesteps}.output \
            2> "$RESULTS_DIR"_err_sample_dif_samp_K_${i}_N_${num_timesteps}.error

        echo "Computing FID: lambda: $lamb and sigma $sigma on dataset_name $dataset_name"          
        python3 fid.py \
                -d Results/ablation_$i/MNIST/sr_operator_gaussian/ddim/K_${i}_N_${num_timesteps}_samples_16eq \
            > "$RESULTS_DIR"_out_fid_samp_K_${i}_N_${num_timesteps}.output \
            2> "$RESULTS_DIR"_err_fid_samp_K_${i}_N_${num_timesteps}.error
    done
mv configs/temp_$i.yaml $RESULTS_DIR1
done

# Running python scripts
# for i in {16,}; do
# sed 's|users/js3006/conddiff/UD2M/Results|users/js3006/conddiff/UD2M/Results_'"${i}"'|g' configs/fine_tuning_lmodel.yaml > configs/temp_$i.yaml
# python3 main.py \
#                 --task $task \
#                 --operator_name "gaussian" \
#                 --lambda_ 0.1 \
#                 --config_file "temp_$i.yaml" \
#                 --dataset_name $dataset_name \
#                 --sigma_model $sigma_model \
#                 --max_unfolded_iter $i \
#                 --kernel_size 3 \
#                 --learned_lambda_sig True \
#                 --sigma_model $sigma_model \
#             > "$RESULTS_DIR"_out_train_dif.output \
#             2> "$RESULTS_DIR"_err_train_dif.error
# mv configs/temp_$i.yaml $RESULTS_DIR1
# done
# Final message
echo "Finish!!"