import inspect

from verl import DataProto
from verl.experimental.reward.reward_loop import register
from verl.utils.reward_score.attacker import compute_score as compute_score_1
from verl.experimental.reward.reward_loop.base import RewardLoopManagerBase
from verl.utils.reward_score import default_compute_score

@register("attacker")
class AttkRewardLoopManager(RewardLoopManagerBase):
    """The reward manager."""

    def __init__(self, config, tokenizer, compute_score=None, reward_router_address=None, reward_model_tokenizer=None):
        super().__init__(config, tokenizer)
        self.compute_score = compute_score_1
        self.is_async_reward_score = inspect.iscoroutinefunction(self.compute_score)
        self.reward_router_address = reward_router_address
        self.reward_model_tokenizer = reward_model_tokenizer

    async def run_single(self, data: DataProto) -> dict:
        assert len(data) == 1, "Only support single data item"
        data_item = data[0]
        response_ids = data_item.batch["responses"]
        response_length = response_ids.shape[-1]
        valid_response_length = data_item.batch["attention_mask"][-response_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]

        ############# INFO ################
        ground_truth = data_item.non_tensor_batch["label"]
        extra_info = data_item.non_tensor_batch.get("extra_info", {})
        question=data_item.non_tensor_batch["question"]
        schema=data_item.non_tensor_batch["schema"]
        

        ############# INFO ################
        tool_extra_fields = data_item.non_tensor_batch.get("tool_extra_fields", None)
        if tool_extra_fields is not None:
            extra_info.update(tool_extra_fields.items())
    
        num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
        rollout_reward_scores = data_item.non_tensor_batch.get("reward_scores", {})
        extra_info["num_turns"] = num_turns
        extra_info["rollout_reward_scores"] = rollout_reward_scores

        response_str = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
        )

        extra_reward_kwargs = (
            {
                "reward_router_address": self.reward_router_address,
                "reward_model_tokenizer": self.reward_model_tokenizer,
            }
            if self.reward_router_address is not None
            else {}
        )
        if self.is_async_reward_score:
            result = await self.compute_score(
                model_output=response_str,
                schema=schema,
                question=question,
                resp_len=len(valid_response_ids)
            )
        else:
            result = await self.loop.run_in_executor(
                None,
                lambda: self.compute_score(
                    model_output=response_str,
                    schema=schema,
                    question=question,
                    resp_len=len(valid_response_ids)
                ),
            )

        reward_extra_info = {}

        score: float
        if isinstance(result, dict):
            score = result["score"]
            for key, value in result.items():
                reward_extra_info[key] = value
        else:
            score = result
            reward_extra_info["acc"] = score

        reward = score

        return {"reward_score": reward, "reward_extra_info": reward_extra_info}