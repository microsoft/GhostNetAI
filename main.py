import os
import yaml
import numpy as np
import typer
from munch import Munch

import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.strategies.ddp import DDPStrategy

from src import algorithms, datasets

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
    gpus = gpus if torch.cuda.is_available() else None
    gpus = [int(i) for i in gpus.split(',')]
 
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
    with open(config) as f:
        conf = Munch(yaml.load(f, Loader=yaml.FullLoader))
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
    dataset = datasets.__dict__[conf.dataset_name](conf=conf)
    learner = algorithms.__dict__[conf.algorithm](conf=conf)

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
        strategy='auto' if accelerator=='cpu' or len(gpus)>1 else 'ddp',
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
            trainer.validate(learner, datamodule=dataset, ckpt_path=evaluate)
        elif predict:
            trainer.predict(learner, datamodule=dataset, ckpt_path=evaluate)
        else:
            trainer.test(learner, datamodule=dataset, ckpt_path=evaluate)
        
    else:
        if conf.weights_init:
            learner.load_state_dict(torch.load(conf.weights_init)['state_dict'], strict=False)
        trainer.fit(learner, datamodule=dataset)

if __name__ == '__main__':
    app()