# Object-segmentation-using-detectron2


 * Detectron2 is FacebookAI's framework for object detection, instance segmentation, and keypoints detection written in PyTorch.

 * Detectron2 makes it convenient to run pre-trained models or train the models from scratch. In this Detectron2 , pre-trained Instance Segmentation, Object detection, keypoints detection, panoptic segmentation.

.* You do not need to download any pre-trained model. The detectron2 takes care of it automatically.

In this project , we worked with images and videos for instance and panoptic segmentation.


# Steps: 

1) Create an new virtual env and activate it python == 3.8.

2) setup your pytorch go to pytorch official page and select it.

I use conda install pytorch torchvision torchaudio cudatoolkit=11.0 -c pytorch (or) use cpu version.

3)Pip install cython        # cython is required for segmentation problems.

4) clone it : git clone https://github.com/facebookresearch/detectron2.git

5) detectron2 will created and change your working dir[ cd detectron2 ].

After that please check Microsoft visual studio will be installed or not.

 Here : https://visualstudio.microsoft.com/visual-cpp-build-tools/

Select only c++ built tools then right corner select >> MSVC v142 - vs 2019 C++ and Windows 10 SDK(10.0....)

6) Back to cmd prompt, type : [pip install -e .].

7) Install opencv > pip install opencv-python.

8) Run main.py if you want change segmentation change it in model_type.

