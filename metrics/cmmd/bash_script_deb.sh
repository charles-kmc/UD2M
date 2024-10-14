#!/bin/bash

# Define the root directory
root_dir="/home/cmk2000/Documents/Years 2/Python codes and results/Python codes/Codes/Diffusion codes with PnP"

# Loop for Inpainting results
for dataset_name in "imagenet_256" "imageffhq_256"; do
    output_file="${dataset_name}_deb.out"
    k_size=(7 25)
    blur_types=("Gaussian" "motion")
    for ii in "${!blur_types[@]}"; do
        blur="${blur_types[$ii]}"
        if [ "$blur" == "Gaussian" ]; then
            blur0="gaussian"
        else
            blur0="$blur"
        fi
        s="${k_size[$ii]}"

        for noise_val in 0.003 0.01; do
            col_name="${dataset_name}_blur_${blur}_noise_${noise_val}"
            
            # Ours
            dir_ours="${root_dir}/PnP_ULA Diffusion/Results_${dataset_name}/Non_blind_Results/eta_1/SGS_rho_Dif_ULA_${blur0}_1.5_Model_diffusion_ffhq_10m_equivariant/26June2024/nfe_3/kenerl_size_${s}/noise_var_${noise_val}/alpha_1.0/Rho_op/c_rho_5.0/Maxiter500"
            dir_ours_true="${dir_ours}/true"
            dir_ours_mmse="${dir_ours}/mmse"
            
            python3 -m main_deb "${dir_ours_true}" "${dir_ours_mmse}" "Ours" "${dataset_name}" "${blur}"  "${noise_val}" >> "${output_file}" 2>&1

            # DiffPIR
            dir_diffpir="${root_dir}/DiffPIR_original/Results_DiffPIR_deb_${dataset_name}/blur_${blur}/Model_name_diffusion_ffhq_10m/28June2024/Noise_level_${noise_val}/Max_teration_500"
            dir_diffpir_true="${dir_diffpir}/true"
            dir_diffpir_mmse="${dir_diffpir}/mmse"
            python3 -m main_deb "${dir_diffpir}/true" "${dir_diffpir}/mmse" "DiffPIR" "${dataset_name}" "${blur}"  "${noise_val}" >> "${output_file}" 2>&1

            # SGS
            dir_sgs="${root_dir}/PnP_ULA Diffusion/Results_SGS_${dataset_name}/Deblurring_Results/MASK_${blur}_1.5_Model_diffusion_ffhq_10m/10July2024/noise_var_${noise_val}/Maxiter100"
            dir_sgs_true="${dir_sgs}/true"
            dir_sgs_mmse="${dir_sgs}/mmse"
            python3 -m main_deb "${dir_sgs_true}" "${dir_sgs_mmse}" "SGS" "${dataset_name}" "${blur}"  "${noise_val}" >> "${output_file}" 2>&1

            # DPS
            dir_dps="${root_dir}/diffusion-posterior-sampling/Results/${dataset_name}/${blur0}_blur/03July2024/${blur0}_blur_size_${s}/Noise_level_${noise_val}/Maxtimestep_1000"
            dir_dps_true="${dir_dps}/true"
            dir_dps_mmse="${dir_dps}/mmse"
            python3 -m main_deb "${dir_dps_true}" "${dir_dps_mmse}" "DPS" "${dataset_name}" "${blur}"  "${noise_val}" >> "${output_file}" 2>&1

            # DPIR IRCNN
            dir_ircnn="${root_dir}/DPIR/Results_DPIR/${dataset_name}/ircnn_color/deblur/${blur0}/kernel_size_${s}/Noise_level_${noise_val}"
            dir_ircnn_true="${dir_ircnn}/true"
            dir_ircnn_mmse="${dir_ircnn}/map"
            python3 -m main_deb "${dir_ircnn_true}" "${dir_ircnn_mmse}" "DPIR IRCNN" "${dataset_name}" "${blur}"  "${noise_val}" >> "${output_file}" 2>&1

            # DPIR DRUNET
            dir_drunet="${root_dir}/DPIR/Results_DPIR/${dataset_name}/drunet_color/deblur/${blur0}/kernel_size_${s}/Noise_level_${noise_val}"
            dir_drunet_true="${dir_drunet}/true"
            dir_drunet_mmse="${dir_drunet}/map"
            python3 -m main_deb "${dir_drunet_true}" "${dir_drunet_mmse}" "DPIR DRUNET" "${dataset_name}" "${blur}"  "${noise_val}" >> "${output_file}" 2>&1
        done
  done
done
