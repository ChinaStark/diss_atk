export WANDB_API_KEY=43a37ea993da69ae3e41784a44b8591809c865d1
# export VLLM_ATTENTION_BACKEND=XFORMERS

DATA_DIR_PATH=/zhiliang/code/Pre-train/data

RUN_ID=3B
GPU_ENV=1GPU
MODEL_ENV=Qwen2.5-Coder-3B-Instruct
# MODEL_P=/zhiliang/huggingface/hub/models--Qwen--Qwen2.5-Coder-7B-Instruct/snapshots/c03e6d358207e414f1eca0bb1891e29f1db0e242
# MODEL_P=/zhiliang/huggingface/hub/models--Qwen--Qwen2.5-Coder-3B-Instruct/snapshots/488639f1ff808d1d3d0ba301aef8c11461451ec5
MODEL_P=/zhiliang/code/Pre-train/logs/Diss_train_3B_less_limit/merged_model_1650steps
PROJECT_NAME=Diss_train_3B_stage2
        
LOG_PATH=logs/$PROJECT_NAME
MODEL_PATH=$MODEL_P
EXPERIMENT_NAME=$GPU_ENV-$MODEL_ENV-$RUN_ID
mkdir -p $LOG_PATH/$EXPERIMENT_NAME
set -x

python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$DATA_DIR_PATH/train_diss_stage_2_RL.parquet \
    data.val_files=$DATA_DIR_PATH/eval.parquet \
    data.train_batch_size=32 \
    data.val_batch_size=16 \
    data.max_prompt_length=8192 \
    data.max_response_length=2048 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False\
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=40 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=40 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.resume_mode=auto \
    trainer.val_before_train=False \
    trainer.default_local_dir=$LOG_PATH/$EXPERIMENT_NAME \
    trainer.default_hdfs_dir=null \
    trainer.resume_mode=auto \
    trainer.save_freq=50 \
    trainer.total_epochs=1 $@ 2>&1 | tee $LOG_PATH/$EXPERIMENT_NAME/grpo.log
# actor_rollout_ref.model.target_modules=[q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj] \
# trainer.resume_mode=auto \
# trainer.resume_from_path=/zhiliang/code/Pre-train/logs/Diss_test/1GPU-Qwen2.5-Coder-7B-Instruct-RL250stage1-7B/data_resume\
# actor_rollout_ref.model.lora_rank=256 \
# actor_rollout_ref.model.lora_alpha=512 \
#
