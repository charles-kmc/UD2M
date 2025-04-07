#/bin/bash

## Deblurring Gaussian noise: 0.003
python3 -m main "/home/cmk2000/Documents/Years 2/Python codes and results/Python codes/Codes/Diffusion codes with PnP/PnP_ULA Diffusion/Results_imageffhq_256/Non_blind_Results/eta_1/SGS_rho_Dif_ULA_gaussian_1.5_Model_diffusion_ffhq_10m_equivariant/26June2024/nfe_3/kenerl_size_7/noise_var_0.01/alpha_1.0/Rho_op/c_rho_5.0/Maxiter500/degraded" "/home/cmk2000/Documents/Years 2/Python codes and results/Python codes/Codes/Diffusion codes with PnP/PnP_ULA Diffusion/Results_imageffhq_256/Non_blind_Results/eta_1/SGS_rho_Dif_ULA_gaussian_1.5_Model_diffusion_ffhq_10m_equivariant/26June2024/nfe_3/kenerl_size_7/noise_var_0.003/alpha_1.0/Rho_op/c_rho_5.0/Maxiter500/true" --batch_size=32 --max_count=30000

