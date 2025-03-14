# ACFD-Net


### Requirements

Tested on 
```
python==3.9.12
torch==1.13.0
h5py==3.6.0
scipy==1.8.0
opencv-python==4.10.0
mmcv==1.4.3
timm=0.6.12
albumentations=1.3.0
tensorboardX==2.5.0
gdown==4.4.0
```

### Downloads
#### Dataset
###### NYU Depth V2

```
$ cd ./datasets
$ wget http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat
$ python ../code/utils/extract_official_train_test_set_from_mat.py nyu_depth_v2_labeled.mat splits.mat ./nyu_depth_v2/official_splits/
```
###### KITTI
Download annotated depth maps data set (14GB) from [[link]](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction) into ./datasets/kitti/data_depth_annotated
```
$ cd ./datasets/kitti/data_depth_annotated/
$ unzip data_depth_annotated.zip
```

Your dataset directory should be
```
root
- nyu_depth_v2
  - bathroom_0001
  - bathroom_0002
  - ...
  - official_splits
- kitti
  - data_depth_annotated
  - raw_data
  - val_selection_cropped
```
#### Pre-trained models
For testing, you can download our [pre-trained models](https://drive.google.com/drive/folders/1hm8HmsjEhYFq0LFq3cIQ7BBn7ECBD8eU?usp=drive_link) and put them in `weights` folder.


### Test

- Test with png images

for NYU Depth V2
```
 $ python ./code/test.py --dataset nyudepthv2 --data_path ./datasets/ --ckpt_dir <path_for_ckpt> --save_eval_pngs  --max_depth 10.0 --max_depth_eval 10.0
  ```
  for KITTI
```
$ python ./code/test.py --dataset kitti --data_path ./datasets/ --ckpt_dir <path_for_ckpt> --save_eval_pngs  --max_depth 80.0 --max_depth_eval 80.0 --kitti_crop [garg_crop or eigen_crop]
  ```
  for other images
```
$ python ./code/test.py --dataset imagepath --data_path <dir_to_imgs>  --save_eval_pngs
  ```

