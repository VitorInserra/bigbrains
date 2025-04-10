#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import sys

def plot_performance_distribution(csv_path='feature_table.csv'):
    try:
        # Read the CSV file
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        sys.exit(1)

    # Check if the performance_metric column exists
    if 'performance_metric' not in df.columns:
        print("Error: 'performance_metric' column not found in the CSV file.")
        sys.exit(1)

    # Extract the performance_metric values
    performance_metric = df['performance_metric']

    # Create the histogram plot
    plt.figure(figsize=(10, 6))
    plt.hist(performance_metric, bins=30, edgecolor='black', alpha=0.7)
    
    # Draw a vertical red dashed line at x=0.5
    plt.axvline(x=performance_metric.mean(), color='red', linestyle='--', linewidth=2, label='Threshold 0.5')
    
    plt.title('Distribution of Performance Metric')
    plt.xlabel('Performance Metric')
    plt.ylabel('Frequency')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    # Save the plot to a file and display it
    plt.savefig("performance_metric_distribution_with_line.png")
    print("Histogram saved as 'performance_metric_distribution_with_line.png'")
    plt.show()

if __name__ == "__main__":
    # Optionally, you can pass the CSV filename as a command line argument
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    else:
        csv_file = 'feature_table.csv'
    plot_performance_distribution(csv_file)
