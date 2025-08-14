from .model import TexTok

def build_model(cfg):
    """Builds the model based on the configuration provided."""
    if cfg.model.name == "textok":
        return TexTok()
    else:
        raise ValueError(f"Model {cfg.model.name} is not supported.")