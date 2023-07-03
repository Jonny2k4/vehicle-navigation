# **Vehicle Navigation**

## **Dataset**

The KITTI Visual Benchmark Suite is a dataset that has been developed specifically for the purpose of benchmarking optical flow, odometry data, object detection, and road/lane detection. The dataset can be downloaded from [here](https://www.cvlibs.net/datasets/kitti/eval_road.php). The road dataset has a dimension of $375$ by $1242$ pixels and contains $600$ individual frames. It is the primary benchmark dataset for road and lane segmentation. This benchmark was developed in partnership with Jannik Fritsch and Tobias Kuehnl of Honda Research Institute Europe GmbH. The road and lane estimate benchmark includes $290$ test and $289$ training images; including three distinct types of road scenes, which are given below. Figure shows some example data (UU, UM, UMM) plotted by MatPlotLib in RGB format.

1. UU - Urban Unmarked
2. UM - Urban Marked
3. UMM - Urban Multiple Marked Lanes

![Alternate text](/readme/kitti_1.png)
![Alternate text](/readme/kitti_2.png)
![Alternate text](/readme/kitti_3.png)

There are $94$ UM, $95$ UMM, and 97 UU images included in the training images. We divided the  into two different sets, one for training and one for validation. The benchmark ranks methods by their maximum F-measure on the Bird’s-eye view (BEV) transformation of the test set. The benchmark also provides LIDAR, stereo and GPS data. In this work, we only made use of the monocular color images and we do not distinguish between the three road categories.

## **Model**

In this repository we implement UNET, U2NET, UNET++, VNET, DNCNN, and MOD-UNET using `Keras-TensorFLow` framework. We also add `keras_unet_collection`(`kuc`) and `segmentation-models`(`sm`) library models which is also implemented using `Keras-TensorFLow`. The following models are available in this repository.

| Model | Name | Reference |
|:---------------|:----------------|:----------------|
| `dncnn`     | DNCNN         | [Zhang et al. (2017)](https://ieeexplore.ieee.org/document/7839189) |
| `unet`      | U-net           | [Ronneberger et al. (2015)](https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28) |
| `vnet`      | V-net (modified for 2-d inputs) | [Milletari et al. (2016)](https://arxiv.org/abs/1606.04797) |
| `unet++` | U-net++         | [Zhou et al. (2018)](https://link.springer.com/chapter/10.1007/978-3-030-00889-5_1) |
| `u2net`     | U^2-Net         | [Qin et al. (2020)](https://arxiv.org/abs/2005.09007) |
| `fapnet`     | FAPNET         | [Samiul et al. (2022)](https://www.mdpi.com/1424-8220/22/21/8245) |
|  | [**keras_unet_collection**](https://github.com/yingkaisha/keras-unet-collection) |  |
| `kuc_r2unet`   | R2U-Net         | [Alom et al. (2018)](https://arxiv.org/abs/1802.06955) |
| `kuc_attunet`  | Attention U-net | [Oktay et al. (2018)](https://arxiv.org/abs/1804.03999) |
|  | [**segmentation-models**](https://github.com/yingkaisha/keras-unet-collection) |  |
| `sm_linknet`     | LINK-Net         | [Chaurasia et al. (2017)](https://arxiv.org/pdf/1707.03718.pdf) |
| `sm_fpn`     | FPS-Net         | [Xiao et al. (2021)](https://arxiv.org/pdf/2103.00738.pdf) |

## **Setup**

First clone the github repo in your local or server machine by following:

```
git clone [https://github.com/samiulengineer/road_segmentation.git](https://github.com/Jonny2k4/vehicle-navigation.git)
```


Change the working directory to project root directory. Use Conda/Pip to create a new environment and install dependency from `requirement.txt` file. The following command will install the packages according to the configuration file `requirement.txt`.

```
pip install -r requirements.txt
```

Keep the above mention dataset in the data folder that give you following structure. Please do not change the directory name `image` and `gt_image`.

```
--data
    --image
        --um_000000.png
        --um_000001.png
            ..
    --gt_image
        --um_road_000000.png
        --um_road_000002.png
            ..
```

## **Experiment**

### **Training**

After setup the required package run the following experiment. The experiment is based on combination of parameters passing through `argparse` and `config.yaml`.

1. An example is given below. Ignore gpu statement if you don't have gpu.
2. During training some new directories will be created; `csv_logger`, `logs`, `model` and `prediction`.
3. You can check the prediction of validation images inside the `prediction > YOUR_MODELNAME > validation > experiment`.
4. You must paste the dataset in the `data` directory inside your `root directory`

```
python project/train.py  \
    --root_dir YOUR_ROOT_DIR \
    --dataset_dir YOUR_ROOT_DIR/data/ \
    --model_name fapnet \
    --epochs 10 \
    --batch_size 3 \
    --gpu YOUR_GPU_NUMBER \
    --experiment road_seg \
```

### **Testing**

Thus, model checkpoint will be saved in the model directory. This checkpoint is required for testing purpose. Run following command to test the model on test dataset.

1. You can check the prediction of test images inside the `prediction > YOUR_MODELNAME > test > experiment`.
2. During training dataset will be divided as training (80%), validation (10%), test (10%)

```
python project/test.py \
    --dataset_dir YOUR_ROOT_DIR/data/ \
    --model_name fapnet \
    --load_model_name MODEL_CHECKPOINT_NAME \
    --experiment road_seg \
    --gpu YOUR_GPU_NUMBER \
    --evaluation False \
```

### **Evaluation on Image**

If you have the images without mask, we need to do data pre-preprocessing before passing to the model checkpoint. In that case, run the following command to evaluate the model without any mask.

1. You can check the prediction of test images inside the `prediction > YOUR_MODELNAME > eval > experiment`.

```
python project/test.py \
    --dataset_dir YOUR_IMAGE_DIR/ \
    --model_name fapnet \
    --load_model_name MODEL_CHECKPOINT_NAME \
    --experiment road_seg \
    --gpu YOUR_GPU_NUMBER \
    --evaluation True \
```

### **Evaluation on Video**

Our model also can predict the road from video data. Run following command for evaluate the model on a video.

```
python project/test.py \
    --video_path PATH_TO_YOUR_VIDEO \
    --model_name fapnet \
    --load_model_name MODEL_CHECKPOINT_NAME \
    --experiment road_seg \
    --gpu YOUR_GPU_NUMBER \
    --evaluation True \
```
