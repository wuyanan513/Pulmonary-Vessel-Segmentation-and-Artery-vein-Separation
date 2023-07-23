
<div align="center">

<samp>

<h2> Transformer-based 3D U-Net for Pulmonary Vessel Segmentation and Artery-vein Separation from CT Images </h1>

<h4> Yanan Wu, Shouliang Qi#, Meihuan Wang, Shuiqing Zhao, Haowen Pang, Jiaxuan Xu, Long Bai, and Hongliang Ren# </h3>

</samp>   

</div>     
    
---

If you find our code or paper useful, please cite as (This paper will be updated later)

```bibtex
@article{wu2022two,
  title={Two-stage Contextual Transformer-based Convolutional Neural Network for Airway Extraction from CT Images},
  author={Wu, Yanan and Zhao, Shuiqing and Qi, Shouliang and Feng, Jie and Pang, Haowen and Chang, Runsheng and Bai, Long and Li, Mengqi and Xia, Shuyue and Qian, Wei and others},
  journal={arXiv preprint arXiv:2212.07651},
  year={2022}
}
@article{wu2023transformer,
  title={Transformer-based 3D U-Net for pulmonary vessel segmentation and artery-vein separation from CT images},
  author={Wu, Yanan and Qi, Shouliang and Wang, Meihuan and Zhao, Shuiqing and Pang, Haowen and Xu, Jiaxuan and Bai, Long and Ren, Hongliang},
  journal={Medical \& Biological Engineering \& Computing},
  pages={1--15},
  year={2023},
  publisher={Springer}
}
```

---
## Abstract
Transformer-based methods have led to the revolutionizing of multiple computer vision tasks. Inspired by this, we propose a transformer-based network with a channel-enhanced attention module to explore contextual and spatial information in non-contrast (NC) and contrast-enhanced (CE) computed tomography (CT) images for pulmonary vessel segmentation and artery-vein separation. Our proposed network employs a 3D contextual transformer module in the encoder and decoder part and a double-attention module in skip connection  to effectively produce high-quality vessel and artery-vein segmentation. Extensive experiments are conducted on the in-house dataset and the ISICDM2021 challenge dataset. The in-house dataset includes 56 NC CT scans with vessel annotations. The challenge dataset consists of 14 NC and 14 CE CT scans with vessel and artery-vein annotations. For vessel segmentation, Dice is 0.840 for CE CT and 0.867 for NC CT. For artery-vein separation, the proposed method achieves a Dice of 0.758 of CE images and 0.602 of NC images. Quantitative and qualitative results demonstrate that the proposed method achieved high accuracy for pulmonary vessel segmentation and artery-vein separation. This method provides useful support for further research associated with the vascular system in CT images.  

<p align="center">
<img src="graph abstract.png" alt="TransformerVessel" width="1000"/>
</p>


---
## Environment

- SimpleITK
- PyTorch
- numpy
- pandas
- scipy
- scikit-learn
- timm
- tqdm
- pickle

## Directory Setup
<!---------------------------------------------------------------------------------------------------------------->
In this project, we implement our method using the Pytorch library, the structure is as follows: 
 
- model_ds.py : 3D CoT module and double attention module.
- main.py : training the model in the provided dataset
- loss.py : the focal loss and dice loss are combined
- utils.py

---
## Dataset
1. The in-house dataset was provided by the hospital. 
2. The ISICDM dataset released in the ISICDM 2021 challenge and labeled by the challenge organizer.

---



## References
Code adopted and modified from:
1. CoTNet model
    - Paper [Contextual Transformer Networks for Visual Recognition](https://arxiv.org/pdf/2107.12292.pdf).
    - official pytorch implementation [Code](https://github.com/JDAI-CV/CoTNet.git).
2. double attention model
    - Paper [A^2-Nets: Double Attention Networks](https://proceedings.neurips.cc/paper_files/paper/2018/file/e165421110ba03099a1c0393373c5b43-Paper.pdf).
    - official pytorch implementation [Code](https://github.com/nguyenvo09/Double-Attention-Network.git).

---

## Contact
For any queries, please raise an issue or contact [Yanan Wu](mailto:yananwu513@gmail.com).

---
