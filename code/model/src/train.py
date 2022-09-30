import copy
import hydra
import pytorch_lightning as pl
import logging
from .models.models import Model
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import RichProgressBar, ModelSummary
from typing import List

# configure lightning logging
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

@hydra.main(config_path='configs', config_name='config', version_base='1.2')
def train(cfg: DictConfig):
    from src import utils

    # fix random seed
    pl.utilities.seed.seed_everything(cfg.seed, workers=True)

    # Add derived cfg (cfgs that are computed from existing cfgs)
    utils.add_derived_cfg_before_init(cfg)

    # --------- init datamodule ---------
    datamodule = hydra.utils.instantiate(cfg.datamodule, data_dir=cfg.data_dir, model_cfg=cfg.model, _recursive_=False)

    # add `id_to_label` (only for NrSpanModel)
    OmegaConf.set_struct(cfg, False)
    cfg.datamodule.id_to_label = getattr(datamodule, 'id_to_label', None)
    OmegaConf.set_struct(cfg, True)

    # --------- init model ---------
    # - you must use deepcopy otherwise function add_derived_cfg_after_init
    # will change the value of cfg
    model = Model(
        datamodule_cfg=copy.deepcopy(cfg.datamodule),
        model_cfg=copy.deepcopy(cfg.model),
        optimizer_cfg=copy.deepcopy(cfg.optimizer),
        trainer_cfg=copy.deepcopy(cfg.trainer))

    # --------- init lightning loggers ---------
    logger = hydra.utils.instantiate(cfg.logger)

    # --------- init lightning callbacks (e.g., early_stop, model_ckpt) ---------
    callbacks: List[pl.Callback] = []
    if "callbacks" in cfg:
        for cb_name, cb_cfg in cfg.callbacks.items():
            if "_target_" in cb_cfg:
                callbacks.append(hydra.utils.instantiate(cb_cfg))
    # callbacks.append(RichProgressBar())  # rich progress bar
    callbacks.append(ModelSummary(max_depth=1))

    # --------- init plugins ---------
    strategy = None
    if cfg.get('strategy'):
        strategy = hydra.utils.instantiate(cfg.strategy)

    # --------- init lightning trainer ---------
    trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=logger,
        strategy=strategy,
        _convert_='partial')
    

    # ------------------------
    # train/test OR find lr
    # ------------------------
    if cfg.mode == 'train':
        # train model
        trainer.fit(model, datamodule)

        # test model
        if cfg.test_after_train:
            trainer.test(ckpt_path='best', datamodule=datamodule)

    elif cfg.mode == 'lr_finder':
        # Plot learning rate
        lr_finder = trainer.tuner.lr_find(model=model, datamodule=datamodule)
        fig = lr_finder.plot(suggest=True)
        fig.show()

    # finalize
    utils.finalize(logger)


    # delimiters
    print('='*40)
    print('='*40)
    print('\n')
