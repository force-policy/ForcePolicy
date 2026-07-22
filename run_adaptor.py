"""
Run data adaptor.

Usage:
    python run_adaptor.py \
           --adaptor [adaptor name] \
           --task_path [task path] \
           --lowdim_name [lowdim_name] \
           --save_lowdim_name [save_lowdim_name] \
           (--data_freq [data frequency, by default 1000]) \
           --tcp_pose_key [tcp pose key, by default 'tcp_pose'] \
           --tcp_vel_key [tcp vel key, by default 'tcp_vel'] \
           --wrench_key [wrench key, by default 'wrench'] \
           --timestamp_key [timestamp key, by default 'timestamp'] \
           --vis_name [vis name, by default None (for no visualization)] \
"""
import argparse

from adaptor import DataAdaptor
from configs import get_config


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "Run data adaptor.")
    parser.add_argument("--adaptor", type = str, required = True, help = "adaptor name")
    parser.add_argument("--task_path", type = str, required = True, help = "task path")
    parser.add_argument("--lowdim_name", type = str, required = True, help = "lowdim_name")
    parser.add_argument("--save_lowdim_name", type = str, required = True, help = "save_lowdim_name")
    parser.add_argument("--data_freq", type = int, default = 1000, help = "data frequency")
    parser.add_argument("--tcp_pose_key", type = str, default = "tcp_pose", help = "tcp pose key")
    parser.add_argument("--tcp_vel_key", type = str, default = "tcp_vel", help = "tcp vel key")
    parser.add_argument("--wrench_key", type = str, default = "wrench", help = "wrench key")
    parser.add_argument("--timestamp_key", type = str, default = "timestamp", help = "timestamp key")
    parser.add_argument("--vis_name", type = str, default = None, help = "vis name")
    parser.add_argument("--vis_mid_step_name", type = str, default = None, help = "visualize intermediate step (Pass 1)")
    parser.add_argument("--specify_seq_name", type = str, default = None, help = "optional per-patch specify sequence file (per scene, under lowdim/), e.g. from scripts/classify_power_source.py. If omitted, the single global config specify is used.")
    args = parser.parse_args()

    adaptor = DataAdaptor(get_config(args.adaptor, "adaptor"))
    adaptor.label_task(
        task_path = args.task_path,
        lowdim_name = args.lowdim_name,
        save_lowdim_name = args.save_lowdim_name,
        data_freq = args.data_freq,
        tcp_pose_key = args.tcp_pose_key,
        tcp_vel_key = args.tcp_vel_key,
        wrench_key = args.wrench_key,
        timestamp_key = args.timestamp_key,
        vis_name = args.vis_name,
        vis_mid_step_name = args.vis_mid_step_name,
        specify_seq_name = args.specify_seq_name
    )
