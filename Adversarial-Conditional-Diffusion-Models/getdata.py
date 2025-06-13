import pandas as pd

root = "/users/js3006/Conditional-Diffusion-Models-for-IVP/Adversarial-Conditional-Diffusion-Models/Results/VarSigma/Results/sr_lora/operator_bicubic/10-06-2025/ddim/Max_iter_1"

T = [1, 3, 9, 27]
K = [1, 3, 9, 27]

CSVfiles = [f"{root}/timesteps_{t}/Unfolded_steps_{k}/zeta_0.9/eta_0.8/lambda_1.0/sigma_0.0125init_prev/metrics_psnr_ssim_lpips_results.csv" for t in T for k in K]

def read_csv_files(files):
    dataframes = []
    for file in files:
        try:
            df = pd.read_csv(file)
            # Add a column to identify the source file
            df["K"] = file.split("Unfolded_steps_")[1].split("/")[0]  # Extract K from the filename
            df["T"] = file.split("timesteps_")[1].split("/")[0]  # Extract T from the filename
            dataframes.append(df)
        except FileNotFoundError:
            print(f"File {file} not found.")
    return dataframes

def concatenate_dataframes(dataframes):
    if not dataframes:
        return pd.DataFrame()  # Return an empty DataFrame if no dataframes are provided
    return pd.concat(dataframes, ignore_index=True)

def main():
    dataframes = read_csv_files(CSVfiles)
    combined_df = concatenate_dataframes(dataframes)
    
    # Save the combined DataFrame to a new CSV file
    output_file = f"{root}/combined_metrics_results.csv"
    combined_df.to_csv(output_file, index=False)
    print(f"Combined data saved to {output_file}")
if __name__ == "__main__":
    main()