# encoding: utf-8
"""
@author:  Ryuk
@contact: jeryuklau@gmail.com
"""

import logging
import os
import sys
import yaml
from collections import OrderedDict

import torch
from torch.nn.parallel import DistributedDataParallel

sys.path.append('.')

from noisexorcist.data import build_dataloader, select_inputs
from noisexorcist.model import build_model
from noisexorcist.loss import build_loss
from noisexorcist.engine import default_argument_parser, default_setup, launch
from noisexorcist.evaluation.testing import flatten_results_dict
from noisexorcist.solver import build_lr_scheduler, build_optimizer
from noisexorcist.utils.checkpoint import Checkpointer, PeriodicCheckpointer
from noisexorcist.evaluation import inference_on_dataset, print_csv_format, SeEvaluator
from noisexorcist.utils import comm, device
from noisexorcist.utils.events import (
    CommonMetricPrinter,
    EventStorage,
    JSONWriter,
    TensorboardXWriter
)

logger = logging.getLogger("noisexorcist")


def get_evaluator(cfg, output_dir=None):
    data_loader = build_dataloader(cfg["DATA"], split="val")
    return data_loader, SeEvaluator(cfg, output_dir)


def do_test(cfg, model):
    logger.info("Prepare testing set")
    try:
        data_loader, evaluator = get_evaluator(cfg)
    except NotImplementedError:
        logger.warn(
            "No evaluator found. implement its `build_evaluator` method."
        )
        raise

    results = inference_on_dataset(cfg, model, data_loader, evaluator)

    if comm.is_main_process():
        assert isinstance(
            results, dict
        ), "Evaluator must return a dict on the main process. Got {} instead.".format(
            results
        )
        logger.info("Evaluation results in csv format")
        print_csv_format(results)

    return results


def do_train(cfg, model, resume=False):
    data_loader = build_dataloader(cfg["DATA"], split="train")

    model.train()
    optimizer = build_optimizer(model, cfg["SOLVER"])

    iters_per_epoch = len(data_loader.dataset) // cfg["DATA"]["TRAIN_BATCH_SIZE"]
    scheduler = build_lr_scheduler(cfg["SOLVER"], optimizer, iters_per_epoch)
    loss = build_loss(cfg["LOSSES"])

    checkpointer = Checkpointer(
        model,
        cfg["OUTPUT_DIR"],
        save_to_disk=comm.is_main_process(),
        optimizer=optimizer,
        **scheduler
    )

    start_epoch = (
            checkpointer.resume_or_load(cfg["MODEL"]["WEIGHTS"], resume=resume).get("epoch", -1) + 1
    )
    iteration = start_iter = start_epoch * iters_per_epoch

    max_epoch = cfg["SOLVER"]["MAX_EPOCH"]
    max_iter = max_epoch * iters_per_epoch
    warmup_iters = cfg["SOLVER"]["WARMUP_ITERS"]
    delay_epochs = cfg["SOLVER"]["DELAY_EPOCHS"]

    periodic_checkpointer = PeriodicCheckpointer(checkpointer, cfg["SOLVER"]["CHECKPOINT_PERIOD"], max_epoch)
    if len(cfg["DATA"]["TESTS"]["METRICS"]) == 1:
        metric_name = cfg["DATA"]["TESTS"]["METRICS"]
    else:
        metric_name = cfg["DATA"]["TESTS"]["METRICS"][0] + "/metric"

    writers = (
        [
            CommonMetricPrinter(max_iter),
            JSONWriter(os.path.join(cfg["OUTPUT_DIR"], "metrics.json")),
            TensorboardXWriter(cfg["OUTPUT_DIR"])
        ]
        if comm.is_main_process()
        else []
    )

    logger.info("Start training from epoch {}".format(start_epoch))
    with EventStorage(start_iter) as storage:
        for epoch in range(start_epoch, max_epoch):
            data_loader_iter = iter(data_loader)
            storage.epoch = epoch
            for i in range(iters_per_epoch):
                data = next(data_loader_iter)
                data = device.to_device(data, cfg["MODEL"]["DEVICE"])
                storage.iter = iteration

                model_inputs = select_inputs(cfg, data)
                model_outputs = model(model_inputs)
                loss_dict = loss(model_outputs, data)
                losses = sum(loss_dict.values())
                assert torch.isfinite(losses).all(), loss_dict

                loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
                losses_reduced = sum(loss for loss in loss_dict_reduced.values())
                if comm.is_main_process():
                    storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()
                storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)

                if iteration - start_iter > 5 and \
                        ((iteration + 1) % 200 == 0 or iteration == max_iter - 1) and \
                        ((iteration + 1) % iters_per_epoch != 0):
                    for writer in writers:
                        writer.write()

                iteration += 1

                if iteration <= warmup_iters:
                    scheduler["warmup_sched"].step()

            # Write metrics after each epoch
            for writer in writers:
                writer.write()

            if iteration > warmup_iters and (epoch + 1) > delay_epochs:
                scheduler["lr_sched"].step()

            if (
                    cfg["DATA"]["TESTS"]["EVAL_PERIOD"] > 0
                    and (epoch + 1) % cfg["DATA"]["TESTS"]["EVAL_PERIOD"] == 0
                    and iteration != max_iter - 1
            ):
                results = do_test(cfg, model)
                # Compared to "train_net.py", the test results are not dumped to EventStorage
            else:
                results = {}
            flatten_results = flatten_results_dict(results)

            metric_dict = dict(metric=flatten_results[metric_name] if metric_name in flatten_results else -1)
            periodic_checkpointer.step(epoch, **metric_dict)


def setup(args):
    """
    Create configs and perform basic setups.
    """
    with open(args.config_file, "r") as file:
        cfg = yaml.safe_load(file)
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    model = build_model(cfg["MODEL"])
    logger.info("Model:\n{}".format(model))

    distributed = comm.get_world_size() > 1
    if distributed:
        model = DistributedDataParallel(
            model.cuda(), device_ids=[comm.get_local_rank()], broadcast_buffers=False
        )

    do_train(cfg, model, resume=args.resume)
    return do_test(cfg, model)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
