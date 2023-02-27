# SegPPC

## Overview
The Pytorch implementation of _Expand Prototype Modalities_




<!-- >Though image-level weakly supervised semantic segmentation (WSSS) has achieved great progress with Class Activation Maps (CAMs) as the cornerstone, the large supervision gap between classification and segmentation still hampers the model to generate more complete and precise pseudo masks for segmentation. In this study, we propose weakly-supervised pixel-to-prototype contrast that can provide pixel-level supervisory signals to narrow the gap. Guided by two intuitive priors, our method is executed across different views and within per single view of an image, aiming to impose cross-view feature semantic consistency regularization and facilitate intra(inter)-class compactness(dispersion) of the feature space. Our method can be seamlessly incorporated into existing WSSS models without any changes to the base networks and does not incur any extra inference burden. Extensive experiments manifest that our method consistently improves two strong baselines by large margins, demonstrating the effectiveness. -->
<img width="801" alt="图片" src="imgs/overview_1.png">


## Prerequisites
- Python 3.6
- pytorch>=1.6.0
- torchvision
- CUDA>=9.0
- pydensecrf from https://github.com/lucasb-eyer/pydensecrf
- others (opencv-python etc.)


## Preparation

1. Clone this repository.
2. Data preparation.
   Download PASCAL VOC 2012 devkit following instructions in [here](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/#devkit. )
   It is suggested to make a soft link toward downloaded dataset. 
   Then download the annotation of VOC 2012 trainaug set (containing 10582 images) from [here](https://www.dropbox.com/s/oeu149j8qtbs1x0/SegmentationClassAug.zip?dl=0) and place them all as ```VOC2012/SegmentationClassAug/xxxxxx.png```. 
   Download the image-level labels ```cls_label.npy``` from [here](https://github.com/YudeWang/SEAM/tree/master/voc12/cls_label.npy) and place it into ```voc12/```, or you can generate it by yourself.
3. Download ImageNet pretrained backbones.
   We use ResNet-38 for initial seeds generation and ResNet-101 for segmentation training. 
   Download pretrained ResNet-38 from [here](https://drive.google.com/file/d/15F13LEL5aO45JU-j45PYjzv5KW5bn_Pn/view).
   The ResNet-101 can be downloaded from [here](https://download.pytorch.org/models/resnet101-5d3b4d8f.pth).
 

## Model Zoo
   Download the trained models and category performance below.
   
   | baseline | model       | train mIoU | val mIoU | test mIoU |   checkpoint (OneDrive)   |       category performance (test)                     |
| -------- | ----------- | :---------: | :-------: | :---------: | :------------: | :----------------------------------------------------------: |
| PPC      | contrast    |    61.5     |   58.4    |      -      | [[download]](https://1drv.ms/u/s!AgGL9MGcRHv0mQSKoJ6CDU0cMjd2?e=dFlHgN) |                                                              |
|          | affinitynet |    69.2     |     -     |             | [[download]](https://1drv.ms/u/s!AgGL9MGcRHv0mQXi0SSkbUc2sl8o?e=AY7AzX) |                                                              |
|          | deeplabv1   |      -      |   67.7*   |    67.4*    | [[download]](https://1drv.ms/u/s!AgGL9MGcRHv0mQgpb3QawPCsKPe9?e=4vly0H) | [[link]](http://host.robots.ox.ac.uk:8080/anonymous/FVG7VK.html) |
| Ours     | contrast    |    64.4     |     61.3     |      -      | [[download]](https://) |                                                              |
|          | deeplabv2   |      -      |   69.0   |    69.6    | [[download]](https://) | [[link]](http://) |

 \* indicates using densecrf.

   The training results including initial seeds, intermediate products and pseudo masks can be found [here](https://).
   
   .. Trained weight and mask will be published soon

## Usage

### Step1: Initial Seed Generation with Contrastive Learning.
1. Contrast train.
   ```
   python contrast_clustering_train.py \
       --session_name $your_session_name \
       --network network.resnet38_contrast_clustering \
       --lr 0.01 --num_workers 8 --train_list voc12/train_aug.txt \
       --weights pretrained/ilsvrc-cls_rna-a1_cls1000_ep-0001.params \
       --voc12_root /home/subin/Datasets/VOC2012/VOCdevkit/VOC2012 \
       --tblog_dir ./tblog --batch_size 8 --max_epoches 8
   ```

2. Contrast inference.

   Train from scratch, set ```--weights``` and then run:
   ```
   python contrast_infer.py \
     --weights $contrast_weight \ 
     --infer_list $[voc12/val.txt | voc12/train.txt | voc12/train_aug.txt] \
     --out_cam $your_cam_npy_dir \
     --out_cam_pred $your_cam_png_dir \
     --out_crf $your_crf_png_dir
   ```

3. Evaluation.

   Following SEAM, we recommend you to use ```--curve``` to select an optimial background threshold.
   ```
   python eval.py \
     --list VOC2012/ImageSets/Segmentation/$[val.txt | train.txt] \
     --predict_dir $your_result_dir \
     --gt_dir VOC2012/SegmentationClass \
     --comment $your_comments \
     --type $[npy | png] \
     --curve True
   ```

<!-- ### Step2: Refine with AffinityNet.
1. Preparation.

   Prepare the files (```la_crf_dir``` and ```ha_crf_dir```) needed for training AffinityNet. You can also use our processed crf outputs with ```alpha=la/ha``` from [here]().
   ```
   python aff_prepare.py \
     --voc12_root VOC2012 \
     --cam_dir $your_cam_npy_dir \
     --out_crf $your_crf_alpha_dir 
   ```

2. AffinityNet train.
   ```
   python aff_train.py \
     --weights $pretrained_model \
     --voc12_root VOC2012 \
     --la_crf_dir $your_la_crf_dir \
     --ha_crf_dir $your_ha_crf_dir \
     --session_name $your_session_name
   ```

3. Random walk propagation & Evaluation.

   Use the trained AffinityNet to conduct RandomWalk for refining the CAMs from Step1. Trained model can be found in Model Zoo.
   ```
   python aff_infer.py \
     --weights $aff_weights \
     --voc12_root VOC2012 \
     --infer_list $[voc12/val.txt | voc12/train.txt] \
     --cam_dir $your_cam_dir \
     --out_rw $your_rw_dir
   ```

4. Pseudo mask generation. 
   Generate the pseudo masks for training the DeepLab Model. Dense CRF is used in this step.
   ```
   python aff_infer.py \
     --weights $aff_weights \
     --infer_list voc12/trainaug.txt \
     --cam_dir $your_cam_dir \
     --voc12_root VOC2012 \
     --out_rw $your_rw_dir
   ```
   
   Pseudo masks of train+aug set can be downloaded here: https://drive.google.com/file/d/1TFw-e6P2tG3AYUgBLTw1pO0NVuBoXi4p/view?usp=sharing.


### Step3: Segmentation training with DeepLab
1. Training. 
   
   we use the segmentation repo from https://github.com/YudeWang/semantic-segmentation-codebase. Training and inference codes are available in ```segmentation/experiment/```. Set ```DATA_PSEUDO_GT: $your_pseudo_label_path``` in ```config.py```. Then run:
   ```
   python train.py
   ```

2. Inference. 

   Check test configration in ```config.py``` (ckpt path, trained model: https://1drv.ms/u/s!AgGL9MGcRHv0mQgpb3QawPCsKPe9?e=4vly0H) and val/test set selection in ```test.py```.  Then run:
   ```
   python test.py
   ```
   
   For test set evaluation, you need to download test set images and submit the segmentation results to the official voc server.
   
For integrating our approach into the [EPS](https://openaccess.thecvf.com/content/CVPR2021/papers/Lee_Railroad_Is_Not_a_Train_Saliency_As_Pseudo-Pixel_Supervision_for_CVPR_2021_paper.pdf) model, you can change branch to ```EPS``` via:
   ```angular2html
   git checkout eps
   ```
Then conduct train or inference following instructions above. Segmentation training follows the same repo in ```segmentation```. Trained models & processed files can be download in Model Zoo. -->

## Acknowledgements
We sincerely thank to the author of PPC [Ye Du](https://github.com/usr922/wseg) who opened their work so we could borrow their codebase for this repository. I would like to express my gratitude to them for their work and to the owner of the original repository that they referenced.

I would also like to express my gratitude to [Mengyang Zhao](https://github.com/masqm/Faster-Mean-Shift) for sharing their codebase for faster mean shift using PyTorch. We have adapted their codebase to enable certain scikit-learn based CPU computations to be performed on a GPU.

Without them, we could not finish this work.
## Citation
<!-- ```
@inproceedings{du2021weakly,
  title={Weakly Supervised Semantic Segmentation by Pixel-to-Prototype Contrast},
  author={Du, Ye and Fu, Zehua and Liu, Qingjie and Wang, Yunhong},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2022}
}
``` -->
