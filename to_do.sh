#0_1 https://cvgl.stanford.edu/projects/uav_data/ から Dataset を　保存，解凍
wget http://vatic2.stanford.edu/stanford_campus_dataset.zip
#0_1_1 保存場所に従い，train.txt,val.txt,test.txtを編集("~/"の部分)
#0_2 requirements.txt に従い，環境構築
pip install -r requirements.txt
#0_3 pseudo-attention作成
env CUDA_VISIBLE_DEVICES=0  python make_pseudo-social-attention.py --d_type train --save
env CUDA_VISIBLE_DEVICES=0  python make_pseudo-social-attention.py --d_type test --save
env CUDA_VISIBLE_DEVICES=0  python make_pseudo-social-attention.py --d_type val --save
env CUDA_VISIBLE_DEVICES=0  python make_pseudo-physical-attention.py --d_type train --save
env CUDA_VISIBLE_DEVICES=0  python make_pseudo-physical-attention.py --d_type test --save
env CUDA_VISIBLE_DEVICES=0  python make_pseudo-physical-attention.py --d_type val --save

#1 social attentionのみ
#1_1 pseudo 使わない場合
env CUDA_VISIBLE_DEVICES=0 python train.py --best_k 20 --not_use_vgg --num_epochs 300 --ge_type traj_rel --gd_type traj_rel --batch_size 32 --l2_loss_type traj_rel --g_learning_rate 0.005 --d_learning_rate 0.005 --checkpoint_num 0 --encoder_h_dim_g 32 --physical_attention_dim 0 --social_attention_type self_attention --social_attention_dim 32  --decoder_h_dim 32 --social_pos_embed --noise_dim 8 --concat_state
#1_2 pseudo 使う場合
env CUDA_VISIBLE_DEVICES=0 python train.py --best_k 20 --not_use_vgg --num_epochs 300 --ge_type traj_rel --gd_type traj_rel --batch_size 32 --l2_loss_type traj_rel --g_learning_rate 0.005 --d_learning_rate 0.005  --d_loss_type bce --checkpoint_num 1 --encoder_h_dim_g 32 --physical_attention_dim 0  --social_attention_type prior --social_attention_dim 32  --decoder_h_dim 32 --social_pos_embed  --noise_dim 8 --concat_state --so_prior_type original --social_tempreture 0.25
env CUDA_VISIBLE_DEVICES=0 python train.py --best_k 20 --not_use_vgg --num_epochs 300 --ge_type traj_rel --gd_type traj_rel --batch_size 32 --l2_loss_type traj_rel --g_learning_rate 0.005 --d_learning_rate 0.005  --d_loss_type bce --checkpoint_num 2 --encoder_h_dim_g 32 --physical_attention_dim 0  --social_attention_type prior --social_attention_dim 32  --decoder_h_dim 32 --social_pos_embed  --noise_dim 8 --concat_state --so_prior_type mul --social_tempreture 0.25

#2 physical attentionも用いる
#2_1 pseudo 使わない場合
env CUDA_VISIBLE_DEVICES=0 python train.py --best_k 20 --not_use_vgg --num_epochs 300 --ge_type traj_rel --gd_type traj_rel --batch_size 4 --l2_loss_type traj_rel --g_learning_rate 0.005 --d_learning_rate 0.005  --d_loss_type bce --checkpoint_num 3 --encoder_h_dim_g 32 --physical_attention_dim 32  --physical_attention_type one_head_attention --social_attention_type prior --social_attention_dim 32  --decoder_h_dim 64 --social_pos_embed  --physical_pos_embed --noise_dim 8 --concat_state --social_tempreture 0.25  --so_prior_type mul 
#2_2 pseudo 使う場合
env CUDA_VISIBLE_DEVICES=0 python train.py --best_k 20 --not_use_vgg --num_epochs 300 --ge_type traj_rel --gd_type traj_rel --batch_size 4 --l2_loss_type traj_rel --g_learning_rate 0.005 --d_learning_rate 0.005  --d_loss_type bce --checkpoint_num 4 --encoder_h_dim_g 32 --physical_attention_dim 32  --physical_attention_type prior --social_attention_type prior --social_attention_dim 32  --decoder_h_dim 64 --social_pos_embed  --physical_pos_embed --noise_dim 8 --concat_state --ph_prior_type nottraj_add --social_tempreture 0.25  --so_prior_type mul

#3 test
env CUDA_VISIBLE_DEVICES=0 python evaluate_model.py --model_num 4_t,4_v

#4 visualize
env CUDA_VISIBLE_DEVICES=0 python visualize_model.py --model_num 4_v