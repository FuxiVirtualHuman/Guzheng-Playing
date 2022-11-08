# A Music-driven Deep Generative Adversarial Model for Guzheng Playing Animation (TVCG 2021)
![在这里插入图片描述](https://img-blog.csdnimg.cn/766c3dcd0bd64586abf4f570e94c9073.png#pic_center)

## Guzheng Playing Dataset
#### Download
You can download the Guzheng playing dataset in Google driven.
The structure of dataset is

 - **guzheng-xxxx.wav:** recorded audio file.
 - **guzheng-xxxx_panoramic.mp4:** recorded MotionCapture video file.
 - **guzheng-xxxx_skeleton.txt:**  recorded animation data.

The audio and corresponding video are not time aligned, so you need to align them if you use the dataset.
#### Visualization
The code depends on python3.8. Run 
```python 
pip install -r requirements.txt
```

To visualize the Guzheng playing animations, run
```python 
python animation_visualization.py --animation_path
```
## Citation
If you use Guzheng playing dataset in your work, please cite

```
@article{chen2021music,
  title={A Music-driven Deep Generative Adversarial Model for Guzheng Playing Animation},
  author={Chen, Jiali and Fan, Changjie and Zhang, Zhimeng and Li, Gongzheng and Zhao, Zeng and Deng, Zhigang and Ding, Yu},
  journal={IEEE Transactions on Visualization and Computer Graphics},
  year={2021},
  publisher={IEEE}
}
```