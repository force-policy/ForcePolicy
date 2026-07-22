import argparse

from adaptor.visualization.label import visualize_labeling


parser = argparse.ArgumentParser(description="Visualize labeling results.")
parser.add_argument("--tcp_file", type = str, help = "Path to the HDF5 file containing TCP poses.")
parser.add_argument("--labeled_file", type = str, help = "Path to the HDF5 file containing labeled frame data.")
parser.add_argument("--save_path", type = str, default = None, help = "Path to save the visualization image. If None, the plot is shown.")
parser.add_argument("--start_ts", type = float, default = None, help = "Start timestamp.")
parser.add_argument("--end_ts", type = float, default = None, help = "End timestamp.")

args = parser.parse_args()

visualize_labeling(args.tcp_file, args.labeled_file, args.start_ts, args.end_ts, args.save_path)
