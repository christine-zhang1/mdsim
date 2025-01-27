from pathlib import Path
import logging
import os

from mdsim.common.registry import registry
from mdsim.trainers.trainer import Trainer


class BaseTask:
    def __init__(self, config):
        self.config = config

    def setup(self, trainer):
        self.trainer = trainer
        # if self.config["checkpoint"] is not None:
        #     self.trainer.load_checkpoint(self.config["checkpoint"])
        # else:
        #     ckpt_dir = (Path(self.trainer.config["cmd"]["checkpoint_dir"]) / 'checkpoint.pt')
        #     if ckpt_dir.exists():
        #         self.trainer.load_checkpoint(ckpt_dir)
            
        # save checkpoint path to runner state for slurm resubmissions
        self.chkpt_path = os.path.join(
            self.trainer.config["cmd"]["checkpoint_dir"], "checkpoint.pt"
        )

    def run(self):
        raise NotImplementedError


@registry.register_task("train")
class TrainTask(BaseTask):
    def _process_error(self, e: RuntimeError):
        e_str = str(e)
        if (
            "find_unused_parameters" in e_str
            and "torch.nn.parallel.DistributedDataParallel" in e_str
        ):
            for name, parameter in self.trainer.model.named_parameters():
                if parameter.requires_grad and parameter.grad is None:
                    logging.warning(
                        f"Parameter {name} has no gradient. Consider removing it from the model."
                    )

    def run(self):
        try:
            self.trainer.train(
                disable_eval_tqdm=self.config.get(
                    "hide_eval_progressbar", False
                )
            )
        except RuntimeError as e:
            self._process_error(e)
            raise e


@registry.register_task("predict")
class PredictTask(BaseTask):
    def run(self):
        assert (
            self.trainer.test_loader is not None
        ), "Test dataset is required for making predictions"
        assert self.config["checkpoint"]
        results_file = "predictions"
        self.trainer.predict(
            self.trainer.test_loader,
            results_file=results_file,
            disable_tqdm=self.config.get("hide_eval_progressbar", False),
        )


@registry.register_task("validate")
class ValidateTask(BaseTask):
    def run(self):
        # Note that the results won't be precise on multi GPUs due to padding of extra images (although the difference should be minor)
        assert (
            self.trainer.val_loader is not None
        ), "Val dataset is required for making predictions"
        assert self.config["checkpoint"]
        self.trainer.validate(
            split="val",
            disable_tqdm=self.config.get("hide_eval_progressbar", False),
        )