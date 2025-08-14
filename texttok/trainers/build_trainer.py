from .textok_trainer import TexTokTrainer


def build_trainer(cfg, model, data_loaders):
        return TexTokTrainer(cfg, model, data_loaders)


            
            
 