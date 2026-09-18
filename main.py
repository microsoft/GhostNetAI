# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import os
import typer

import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.strategies.ddp import DDPStrategy

from src import algorithms, datasets
from src.checkpoints import load_weights
from src.config import load_config, select_component

app = typer.Typer(pretty_exceptions_short=True, pretty_exceptions_show_locals=False)

@app.command()
def main(config:str='./configs/ghostset_de_plain_111824.yaml',
         project:str='GhostNet_DE',
         accelerator:str='gpu', 
         gpus:str='0', 
         logger_type:str='tensorboard',
         session:int=0,
         np_threads:int=32,
         evaluate:str=None,
         val:bool=False,
         predict:bool=False,
         predict_root:str=False,
         seed:int=0):

    ############
    # Set gpus #
    ############
    gpus = [int(i) for i in gpus.split(',')] if accelerator == 'gpu' else []
    if accelerator == 'gpu' and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; use --accelerator cpu")
 
    #############################
    # Set environment variables #
    #############################
    # Set numpy threads
    os.environ["OMP_NUM_THREADS"] = str(np_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(np_threads)
    os.environ["MKL_NUM_THREADS"] = str(np_threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(np_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(np_threads)
    
    #######################
    # Load configurations #
    #######################
    conf = load_config(config)
    conf.session = session
    conf.evaluate = evaluate
    conf.val = val
    conf.predict = predict
    conf.predict_root = predict_root
    if len(gpus) > 1:
        conf.batch_size = int(conf.batch_size / torch.cuda.device_count())

    L.seed_everything(seed)

    torch.set_float32_matmul_precision('medium')

    ###########################
    # Load data and algorithm #
    ###########################
    dataset_class = select_component(datasets.DATASETS, conf.dataset_name, "dataset")
    algorithm_class = select_component(algorithms.ALGORITHMS, conf.algorithm, "algorithm")
    learner = algorithm_class(conf=conf)
    if evaluate is not None:
        load_weights(learner, evaluate)
    elif conf.weights_init:
        result = load_weights(learner, conf.weights_init, strict=False)
        if result.missing_keys or result.unexpected_keys:
            print(f"Initial weights: missing keys={result.missing_keys}, "
                  f"unexpected keys={result.unexpected_keys}")
    dataset = dataset_class(conf=conf)

    ###############
    # Load logger #
    ###############
    if logger_type == 'csv':
        logger = CSVLogger(
            save_dir='./log/{}'.format(conf.algorithm),
            prefix=project,
            name='{}_{}'.format(conf.algorithm, conf.conf_id),
            version=session
        )
    elif logger_type == 'tensorboard':
        logger = TensorBoardLogger(
            save_dir='./log/{}'.format(conf.algorithm),
            prefix=project,
            name='{}_{}'.format(conf.algorithm, conf.conf_id),
            version=session
        )
    else:
        raise ValueError("logger_type must be csv or tensorboard")

    ##################
    # Load callbacks #
    ##################
    checkpoint_callback = ModelCheckpoint(
        monitor='valid_mean_IoU', mode='max', dirpath='./weights/{}/{}'.format(project, conf.algorithm), save_top_k=1,
        filename='{}-{}'.format(conf.conf_id, session) + '-{epoch:02d}-{valid_mean_IoU:.2f}', verbose=True
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')

    #################
    # Setup trainer #
    #################
    trainer = L.Trainer(
        max_steps=conf.num_iters,
        check_val_every_n_epoch=1, 
        log_every_n_steps = conf.log_interval, 
        accelerator=accelerator,
        devices=gpus if accelerator=='gpu' else 'auto',
        # accelerator='cpu',
        logger=None if evaluate is not None else logger,
        callbacks=[lr_monitor, checkpoint_callback],
        # strategy=DDPStrategy(find_unused_parameters=False),
        # strategy='auto',
        strategy='auto',
        num_sanity_val_steps=1,
        # profiler='simple',
        # precision=16,
        enable_progress_bar=True,
    )

    #######
    # RUN #
    #######
    if evaluate is not None:
        if val:
            trainer.validate(learner, datamodule=dataset)
        elif predict:
            trainer.predict(learner, datamodule=dataset, return_predictions=False)
        else:
            trainer.test(learner, datamodule=dataset)
        
    else:
        trainer.fit(learner, datamodule=dataset)

if __name__ == '__main__':
    app()