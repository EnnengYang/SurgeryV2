# Surgery V2
A repository of **'[SurgeryV2: Bridging the Gap Between Model Merging and Multi-Task Learning with Deep Representation Surgery]()'**.


## Abstract
>

<center>
<img src="./surgeryv2.png" alt="Surgery V2" width="800"/>
</center>

## Citation
If you find our paper or this resource helpful, please consider cite:
```

```
Thanks!

## Datasets
Refer to dataset processing in the [task_vectors](https://github.com/mlfoundations/task_vectors).

Or you can download the processed data from [Baidu Cloud disk](https://pan.baidu.com/s/1w0Z2UVv3NVmqDhjH8WTOJQ?pwd=kvg6) or [HugggingFace](https://huggingface.co/collections/tanganke/image-classification-datasets-662abda7d75efe6b0e6b43da).


## Task Vectors / Checkpoints

You can download the fine-tuned checkpoints from the [task_vectors#checkpoints](https://github.com/mlfoundations/task_vectors#checkpoints).
The Google Drive folder is: [task_vectors_checkpoints](https://drive.google.com/drive/folders/1u_Tva6x0p6oxu5Eo0ZZsf-520Cc_3MKw)


*Note: When using ```torch.load(xxx_checkpoint).state_dict()``` fails, you can try ```pickle.load(open(xxx_checkpoint, 'rb')).state_dict()```.*


## Train

- Model Merging Methods (e.g., Weight Averaging, Task Arithmetic, Ties-Merging, AdaMerging)
```
python src/main_tv.py
```

- Model Merging Methods with [Surgery](https://github.com/EnnengYang/RepresentationSurgery) (e.g., Weight Averaging w/ Surgery, Task Arithmetic w/ Surgery, Ties-Merging w/ Surgery, AdaMerging w/ Surgery)

```
python src/main_tv_surgery_v1.py
```

- Model Merging Methods with our [SurgeryV2](https://github.com/EnnengYang/SurgeryV2) (e.g., Weight Averaging w/ Surgery V2, Task Arithmetic w/ Surgery V2, Ties-Merging w/ Surgery V2, AdaMerging w/ Surgery V2)
```
python src/main_tv_surgery_v2.py
```

*Note: Due to machine memory limitations, our implementation reloaded the dataset at each step, which resulted in a significant amount of additional time. If your machine has enough memory, you can load all the data before optimizing the surgery module, which will speed up the training significantly.*


## Acknowledgement
Our implementation references the code below, thanks to them.

- RepresentationSurgery: https://github.com/EnnengYang/RepresentationSurgery

- AdaMerging: https://github.com/EnnengYang/AdaMerging

- Task Arithmetic: https://github.com/mlfoundations/task_vectors

- TIES-MERGING: https://github.com/prateeky2806/ties-merging/tree/main

- Model Soups: https://github.com/mlfoundations/model-soups
