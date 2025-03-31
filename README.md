# [PointPillars: Fast Encoders for Object Detection from Point Clouds](https://arxiv.org/abs/1812.05784) 

A Simple PointPillars PyTorch Implementation for 3D Lidar(KITTI) Detection. [[Zhihu](https://zhuanlan.zhihu.com/p/521277176)]

- It can be run without installing [Spconv](https://github.com/traveller59/spconv), [mmdet](https://github.com/open-mmlab/mmdetection) or [mmdet3d](https://github.com/open-mmlab/mmdetection3d). 
- Only one detection network (PointPillars) was implemented in this repo, so the code may be more easy to read. 
- Sincere thanks for the great open-source architectures [mmcv](https://github.com/open-mmlab/mmcv), [mmdet](https://github.com/open-mmlab/mmdetection) and [mmdet3d](https://github.com/open-mmlab/mmdetection3d), which helps me to learn 3D detection and implement this repo.

## News

- **2025-02** Making PointPillars a python package out of the code is supported.
- **2024-04** Exporting PointPillars to ONNX & TensorRT is supported on branch [feature/deployment](https://github.com/zhulf0804/PointPillars/tree/feature/deployment).

    ![](./figures/pytorch_trt.png)

## Performance

### mAP on KITTI validation set (Easy, Moderate, Hard)

| Repo | Metric | Overall | Pedestrian | Cyclist | Car |
| :---: | :---: | :---: | :---: | :---: | :---: |
| this repo | 3D-BBox | 73.3259 62.7834 59.6278 | 51.4642 47.9446 43.8040 | 81.8677 63.6617 60.9126 | 86.6456 76.7439 74.1668 | 
| [mmdet3d v0.18.1](https://github.com/open-mmlab/mmdetection3d/tree/v0.18.1) | 3D-BBox  | 72.0537, 60.1114, 55.8320 | 52.0263, 46.4037, 42.4841 | 78.7231, 59.9526, 57.2489 | 85.4118, 73.9780, 67.7630 |
| this repo | BEV | 77.8540 69.8003 66.6699 | 59.1687 54.3456 50.5023 | 84.4268 67.1409 63.7409 | 89.9664 87.9145 85.7664 | 
| [mmdet3d v0.18.1](https://github.com/open-mmlab/mmdetection3d/tree/v0.18.1) | BEV | 76.6485, 67.7609, 64.5605 | 59.0778, 53.3638, 48.4230 | 80.9328, 63.3447, 60.0618 | 89.9348, 86.5743, 85.1967 |
| this repo | 2D-BBox | 80.5097 74.6120 71.4758 | 64.6249 61.4201 57.5965 | 86.2569 73.0828 70.1726 | 90.6471 89.3330 86.6583 |
| [mmdet3d v0.18.1](https://github.com/open-mmlab/mmdetection3d/tree/v0.18.1) | 2D-BBox | 78.4938, 73.4781, 70.3613 | 62.2413, 58.9157, 55.3660 | 82.6460, 72.3547, 68.4669 | 90.5939, 89.1638, 87.2511 |
| this repo | AOS | 74.9647 68.1712 65.2817 | 49.3777 46.7284 43.8352 | 85.0412 69.1024 66.2801 | 90.4752 88.6828 85.7298 |
| [mmdet3d v0.18.1](https://github.com/open-mmlab/mmdetection3d/tree/v0.18.1) | AOS | 72.41, 66.23, 63.55 | 46.00, 43.22, 40.94 | 80.85, 67.20, 63.63 | 90.37, 88.27, 86.07 |

- **Note: Here, we report [mmdet3d v0.18.1](https://github.com/open-mmlab/mmdetection3d/tree/v0.18.1) (2022/02/09-2022/03/01) performance based on the officially provided [checkpoint](https://github.com/open-mmlab/mmdetection3d/tree/v0.18.1/configs/pointpillars#kitti). Much improvements were made in the [mmdet3d v1.0.0rc1](https://github.com/open-mmlab/mmdetection3d/tree/v1.0.0rc1)**. 

## Detection Visualization

![](./figures/pc_pred_000134.png)
![](./figures/img_3dbbox_000134.png)

## [Install] 

Install PointPillars as a python package and all its dependencies as follows:

```
cd PointPillars/
pip install -r requirements.txt
python setup.py build_ext --inplace
pip install .
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Pipeline Usage

The project includes a unified pipeline script (`pipeline.py`) that provides a convenient interface for all operations. Here's how to use it:

### Basic Usage
```bash
python pipeline.py --data_root your_path_to_kitti --action <action_name>
```

Available actions:
- `prepare`: Prepare the KITTI dataset
- `reduce_beams`: Reduce the number of LiDAR beams
- `visualize_beams`: Visualize the reduced beams
- `train`: Train the model
- `evaluate`: Evaluate the model
- `test`: Test the model on a single point cloud
- `test_reduced`: Test the model using reduced LiDAR beams
- `full_pipeline`: Run the complete pipeline (prepare → reduce beams → visualize → train → evaluate)

### Examples

1. Prepare the dataset:
```bash
python pipeline.py --data_root your_path_to_kitti --action prepare
```

2. Reduce and visualize LiDAR beams:
```bash
python pipeline.py --data_root your_path_to_kitti --action reduce_beams
python pipeline.py --data_root your_path_to_kitti --action visualize_beams
```

3. Train the model:
```bash
python pipeline.py --data_root your_path_to_kitti --action train
```

4. Evaluate the model:
```bash
python pipeline.py --data_root your_path_to_kitti --action evaluate --ckpt pretrained/epoch_160.pth
```

5. Test on a single point cloud (original beams):
```bash
python pipeline.py --data_root your_path_to_kitti --action test \
                   --pc_path dataset/demo_data/val/000134.bin \
                   --calib_path dataset/demo_data/val/000134.txt \
                   --img_path dataset/demo_data/val/000134.png \
                   --gt_path dataset/demo_data/val/000134_gt.txt
```

6. Test on a single point cloud (reduced beams):
```bash
python pipeline.py --data_root your_path_to_kitti --action test_reduced \
                   --pc_path dataset/demo_data/val/000134.bin \
                   --calib_path dataset/demo_data/val/000134.txt \
                   --img_path dataset/demo_data/val/000134.png \
                   --gt_path dataset/demo_data/val/000134_gt.txt
```

7. Run the complete pipeline:
```bash
python pipeline.py --data_root your_path_to_kitti --action full_pipeline
```

### Notes
- The `test_reduced` action automatically uses the reduced LiDAR beams from the `velodyne_reduced` directory
- The `full_pipeline` action runs all steps in sequence: dataset preparation, beam reduction, visualization, training, and evaluation
- Make sure to run `reduce_beams` before using `test_reduced` or `full_pipeline`

## Dataset Preparation

1. Download KITTI Dataset:
    - [Point cloud data](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_velodyne.zip) (29GB)
    - [Images](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip) (12GB)
    - [Calibration files](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_calib.zip) (16MB)
    - [Labels](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip) (5MB)

    Organize the dataset as follows:
    ```
    kitti
        |- training
            |- calib (#7481 .txt)
            |- image_2 (#7481 .png)
            |- label_2 (#7481 .txt)
            |- velodyne (#7481 .bin)
        |- testing
            |- calib (#7518 .txt)
            |- image_2 (#7518 .png)
            |- velodyne (#7518 .bin)
    ```

2. Pre-process the KITTI dataset:
    ```bash
    python prepare_kitti_dataset.py --data_root your_path_to_kitti
    ```

    This will generate additional files:
    ```
    kitti
        |- training
            |- velodyne_reduced (#7481 .bin)
        |- testing
            |- velodyne_reduced (#7518 .bin)
        |- kitti_gt_database (#19700 .bin)
        |- kitti_infos_train.pkl
        |- kitti_infos_val.pkl
        |- kitti_infos_trainval.pkl
        |- kitti_infos_test.pkl
        |- kitti_dbinfos_train.pkl
    ```

## Usage

### Training
```bash
python train.py --data_root your_path_to_kitti
```

### Evaluation
```bash
python evaluate.py --ckpt pretrained/epoch_160.pth --data_root your_path_to_kitti 
```

### Testing and Visualization

1. Point cloud detection visualization:
```bash
python test.py --ckpt pretrained/epoch_160.pth --pc_path your_pc_path 
```

2. Point cloud detection with ground truth:
```bash
python test.py --ckpt pretrained/epoch_160.pth --pc_path your_pc_path --calib_path your_calib_path --gt_path your_gt_path
```

3. Point cloud and image detection visualization:
```bash
python test.py --ckpt pretrained/epoch_160.pth --pc_path your_pc_path --calib_path your_calib_path --img_path your_img_path
```

### Example Usage

1. Inference on validation set sample (000134):
```bash
python test.py --ckpt pretrained/epoch_160.pth --pc_path dataset/demo_data/val/000134.bin
```

2. Inference with ground truth and image visualization:
```bash
python test.py --ckpt pretrained/epoch_160.pth --pc_path dataset/demo_data/val/000134.bin \
               --calib_path dataset/demo_data/val/000134.txt --img_path dataset/demo_data/val/000134.png \
               --gt_path dataset/demo_data/val/000134_gt.txt
```

3. Inference on test set sample (000002):
```bash
python test.py --ckpt pretrained/epoch_160.pth --pc_path dataset/demo_data/test/000002.bin
```

## Acknowledgments

Thanks for the open source code [mmcv](https://github.com/open-mmlab/mmcv), [mmdet](https://github.com/open-mmlab/mmdetection) and [mmdet3d](https://github.com/open-mmlab/mmdetection3d).
