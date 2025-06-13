# Code to plot a histogram using pandas to load a csv file with single column "L2_distance"
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
def plot_histogram(csv_file, output_image, column, scale = "log"):
    # Load the CSV file
    df = pd.read_csv(csv_file)

    # Check if the column 'L2_distance' exists
    if 'L2_distance' not in df.columns:
        raise ValueError("The CSV file must contain a column named 'L2_distance'.")

    # Plot the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(df[column], bins=50, color='blue', alpha=0.7)
    plt.title(f'Histogram of {column}')
    plt.yscale(scale)  # Set the y-axis scale to log if specified
    plt.xlabel(r'$\Delta W$')
    plt.ylabel('Frequency')
    
    # Save the plot to an image file
    plt.savefig(output_image)
    plt.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Plot histogram from CSV file.')
    parser.add_argument('--csv_file', type=str, help='Path to the input CSV file.')
    parser.add_argument('--output_image', type=str, help='Path to save the output histogram image.')
    parser.add_argument('--column', type=str, default='L2_distance', help='Column name to plot from the CSV file.')
    
    args = parser.parse_args()
    
    plot_histogram(args.csv_file, args.output_image,args.column)