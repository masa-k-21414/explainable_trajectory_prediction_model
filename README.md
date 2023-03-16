# Explainable_Trajectory_Prediction_Model

## Abstract
Recent advances in deep learning have led to the application of computer vision technology. For such practical and safety-critical applications, it is important to be able to explain the rationale behind the computer's decisions in addition to guaranteeing its performance. We deal with trajectory prediction models, a type of action prediction model. Trajectory prediction is a technology that can be applied not only to behavior prediction, but also to behavior analysis and tracking. In recent years, the use of deep learning has improved the accuracy of trajectory prediction, but at the same time, it has inherited the nature of deep models that are unclear in their interpretation. Existing models predict natural trajectories by learning two interactions: interactions from surrounding people (person-person interactions) and interactions from the surrounding environment (person-space interactions). We visualize these interactions using a attention module to visualize the basis for the trajectory prediction model. Although there are existing methods that use the attention module, they are learned without any constraints on attention, and thus produce results that are difficult to interpret. In this study, as shown, by conditioning attention using pseudoattention in which human knowledge is quantified in advance, we stabilize attention estimation and at the same time, obtain a basis for the decision that can be interpreted from the human point of view.

## Data Preparation
SDD raw data comes from [Stanford Drone Dataset](https://cvgl.stanford.edu/projects/uav_data/).<br>
Edit train.txt, val.txt, and test.txt according to the directory where the data was saved.

## Requirements
The codes are developed with python 3.6.9.
Additional packages used are included in the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

## Precomputation Of Pseudo Attention
In our model, pseudo-attention must be pre-created.
Social Pseudo Attention:
```bash
env CUDA_VISIBLE_DEVICES=0  python make_pseudo-social-attention.py --d_type train --save
env CUDA_VISIBLE_DEVICES=0  python make_pseudo-social-attention.py --d_type test --save
env CUDA_VISIBLE_DEVICES=0  python make_pseudo-social-attention.py --d_type val --save
```
Physical Pseudo Attention:
```bash
env CUDA_VISIBLE_DEVICES=0  python make_pseudo-physical-attention.py --d_type train --save
env CUDA_VISIBLE_DEVICES=0  python make_pseudo-physical-attention.py --d_type test --save
env CUDA_VISIBLE_DEVICES=0  python make_pseudo-physical-attention.py --d_type val --save
```

## Model Training

### Use Only Social Attention Module

Learning when using conventional attention module without pseudo attention:
```bash
env CUDA_VISIBLE_DEVICES=0 python train.py --best_k 20 --not_use_vgg --num_epochs 300 --ge_type traj_rel --gd_type traj_rel --batch_size 32 --l2_loss_type traj_rel --g_learning_rate 0.005 --d_learning_rate 0.005 --checkpoint_num 0 --encoder_h_dim_g 32 --physical_attention_dim 0 --social_attention_type self_attention --social_attention_dim 32  --decoder_h_dim 32 --social_pos_embed --noise_dim 8 --concat_state
```
Learning our proposed model using pseudoattention:
```bash
env CUDA_VISIBLE_DEVICES=0 python train.py --best_k 20 --not_use_vgg --num_epochs 300 --ge_type traj_rel --gd_type traj_rel --batch_size 32 --l2_loss_type traj_rel --g_learning_rate 0.005 --d_learning_rate 0.005  --d_loss_type bce --checkpoint_num 1 --encoder_h_dim_g 32 --physical_attention_dim 0  --social_attention_type prior --social_attention_dim 32  --decoder_h_dim 32 --social_pos_embed  --noise_dim 8 --concat_state --so_prior_type original --social_tempreture 0.25
env CUDA_VISIBLE_DEVICES=0 python train.py --best_k 20 --not_use_vgg --num_epochs 300 --ge_type traj_rel --gd_type traj_rel --batch_size 32 --l2_loss_type traj_rel --g_learning_rate 0.005 --d_learning_rate 0.005  --d_loss_type bce --checkpoint_num 2 --encoder_h_dim_g 32 --physical_attention_dim 0  --social_attention_type prior --social_attention_dim 32  --decoder_h_dim 32 --social_pos_embed  --noise_dim 8 --concat_state --so_prior_type mul --social_tempreture 0.25
```

### Use Social And Physical Attention Modules
Learning when using conventional attention module without pseudo attention:
```bash
env CUDA_VISIBLE_DEVICES=0 python train.py --best_k 20 --not_use_vgg --num_epochs 300 --ge_type traj_rel --gd_type traj_rel --batch_size 4 --l2_loss_type traj_rel --g_learning_rate 0.005 --d_learning_rate 0.005  --d_loss_type bce --checkpoint_num 3 --encoder_h_dim_g 32 --physical_attention_dim 32  --physical_attention_type one_head_attention --social_attention_type prior --social_attention_dim 32  --decoder_h_dim 64 --social_pos_embed  --physical_pos_embed --noise_dim 8 --concat_state --social_tempreture 0.25  --so_prior_type mul 
```
Learning our proposed model using pseudoattention:
```bash
env CUDA_VISIBLE_DEVICES=0 python train.py --best_k 20 --not_use_vgg --num_epochs 300 --ge_type traj_rel --gd_type traj_rel --batch_size 4 --l2_loss_type traj_rel --g_learning_rate 0.005 --d_learning_rate 0.005  --d_loss_type bce --checkpoint_num 4 --encoder_h_dim_g 32 --physical_attention_dim 32  --physical_attention_type prior --social_attention_type prior --social_attention_dim 32  --decoder_h_dim 64 --social_pos_embed  --physical_pos_embed --noise_dim 8 --concat_state --ph_prior_type nottraj_add --social_tempreture 0.25  --so_prior_type mul
```

## Evaluation
For example, to evaluate model "4":
```bash
env CUDA_VISIBLE_DEVICES=0 python evaluate_model.py --model_num 4_v
```

## visualization
For example, to visualize the results of model "4":
```bash
env CUDA_VISIBLE_DEVICES=0 python visualize_model.py --model_num 4_v
```
