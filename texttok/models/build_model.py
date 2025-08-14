from .model import TexTokVAE

def build_model(cfg):
    """Builds the model based on the configuration provided."""
    if cfg.model.name == "textok":
        return TexTokVAE()
    else:
        raise ValueError(f"Model {cfg.model.name} is not supported.")