from pathlib import Path

import hydra
import torch
from hydra import compose, initialize_config_dir


def test_c4_equivariance() -> None:
    torch.manual_seed(7)
    with initialize_config_dir(
        version_base=None, config_dir=str(Path("conf").resolve())
    ):
        cfg = compose(config_name="config", overrides=["model=convnext_c4"])
    model = hydra.utils.instantiate(cfg.model.module).eval()
    image = torch.randn(1, 3, 224, 224)
    with torch.inference_mode():
        reference = model.forward_features(image)
        embedding = model.readout(reference).reshape(1, 4, -1)
        assert reference.shape == (1, 4, 192, 7, 7)
        assert not torch.allclose(embedding, embedding.roll(-1, 1))
        for turns in (1, 2, 3):
            observed = model.forward_features(image.rot90(turns, (-2, -1)))
            expected = reference.rot90(turns, (-2, -1)).roll(-turns, 1)
            torch.testing.assert_close(observed, expected, atol=2e-5, rtol=2e-4)
            torch.testing.assert_close(
                model.readout(observed).reshape(1, 4, -1),
                embedding.roll(-turns, 1),
                atol=2e-5,
                rtol=2e-4,
            )
