#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import subprocess
import sys
import os
from typing import Optional, List, Dict

class PointPillarsPipeline:
    def __init__(self, data_root: str, ckpt_path: Optional[str] = None):
        self.data_root = data_root
        self.ckpt_path = ckpt_path or "pretrained/epoch_160.pth"
        
    def prepare_dataset(self) -> bool:
        """Prepare the KITTI dataset for training."""
        print("Preparing KITTI dataset...")
        cmd = f"python prepare_kitti_dataset.py --data_root {self.data_root}"
        return self._run_command(cmd)
    
    def reduce_beams(self) -> bool:
        """Reduce the number of LiDAR beams in the dataset."""
        print("Reducing LiDAR beams...")
        cmd = f"python reduce_lidar_beams.py --data_root {self.data_root}"
        return self._run_command(cmd)
    
    def visualize_reduced_beams(self) -> bool:
        """Visualize the reduced LiDAR beams."""
        print("Visualizing reduced beams...")
        cmd = f"python visualize_reduced_beams.py --data_root {self.data_root}"
        return self._run_command(cmd)
    
    def train(self) -> bool:
        """Train the PointPillars model."""
        print("Training PointPillars model...")
        cmd = f"python train.py --data_root {self.data_root}"
        return self._run_command(cmd)
    
    def evaluate(self) -> bool:
        """Evaluate the trained model."""
        print("Evaluating model...")
        cmd = f"python evaluate.py --ckpt {self.ckpt_path} --data_root {self.data_root}"
        return self._run_command(cmd)
    
    def test(self, pc_path: str, calib_path: Optional[str] = None, 
             img_path: Optional[str] = None, gt_path: Optional[str] = None,
             use_reduced_beams: bool = False) -> bool:
        """Test the model on a single point cloud file."""
        print(f"Testing on {pc_path}...")
        if use_reduced_beams:
            # Replace the original point cloud path with the reduced beams version
            pc_dir = os.path.dirname(pc_path)
            pc_name = os.path.basename(pc_path)
            pc_path = os.path.join(pc_dir, "velodyne_reduced", pc_name)
            print(f"Using reduced beams from: {pc_path}")
            
        cmd = f"python test.py --ckpt {self.ckpt_path} --pc_path {pc_path}"
        
        if calib_path:
            cmd += f" --calib_path {calib_path}"
        if img_path:
            cmd += f" --img_path {img_path}"
        if gt_path:
            cmd += f" --gt_path {gt_path}"
            
        return self._run_command(cmd)
    
    def _run_command(self, cmd: str) -> bool:
        """Run a shell command and return True if successful."""
        try:
            subprocess.run(cmd, shell=True, check=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error running command: {cmd}")
            print(f"Error: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(description="PointPillars Pipeline")
    parser.add_argument("--data_root", required=True, help="Path to KITTI dataset root directory")
    parser.add_argument("--ckpt", default="pretrained/epoch_160.pth", help="Path to checkpoint file")
    parser.add_argument("--action", required=True, choices=[
        "prepare", "reduce_beams", "visualize_beams", "train", "evaluate", "test",
        "test_reduced", "full_pipeline"
    ], help="Action to perform")
    parser.add_argument("--pc_path", help="Path to point cloud file for testing")
    parser.add_argument("--calib_path", help="Path to calibration file for testing")
    parser.add_argument("--img_path", help="Path to image file for testing")
    parser.add_argument("--gt_path", help="Path to ground truth file for testing")
    
    args = parser.parse_args()
    
    pipeline = PointPillarsPipeline(args.data_root, args.ckpt)
    
    actions = {
        "prepare": pipeline.prepare_dataset,
        "reduce_beams": pipeline.reduce_beams,
        "visualize_beams": pipeline.visualize_reduced_beams,
        "train": pipeline.train,
        "evaluate": pipeline.evaluate,
        "test": lambda: pipeline.test(args.pc_path, args.calib_path, args.img_path, args.gt_path),
        "test_reduced": lambda: pipeline.test(args.pc_path, args.calib_path, args.img_path, args.gt_path, use_reduced_beams=True),
        "full_pipeline": lambda: all([
            pipeline.prepare_dataset(),
            pipeline.reduce_beams(),
            pipeline.visualize_reduced_beams(),
            pipeline.train(),
            pipeline.evaluate()
        ])
    }
    
    success = actions[args.action]()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 