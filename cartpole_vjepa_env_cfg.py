# POC : Cartpole vu par caméra, avec un encodeur V-JEPA 2 gelé comme extracteur de features.
# Lien symbolique dans source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/cartpole/
# Prérequis : uv pip install "transformers>=4.53"

import gymnasium as gym
import torch
import torch.nn.functional as F

import isaaclab.envs.mdp as mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from .cartpole_camera_env_cfg import CartpoleRGBCameraEnvCfg

HF_ID = "facebook/vjepa2-vitl-fpc64-256"  # ViT-L, ~300M params, sortie 1024-d
RES = 224  # baisse à 160 ou 128 si c'est trop lent
DEVICE = "cuda:0"

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)


def _load_vjepa(*args, **kwargs):
    from transformers import AutoModel

    model = AutoModel.from_pretrained(HF_ID, torch_dtype=torch.bfloat16)
    model = model.to(DEVICE).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"[V-JEPA] {HF_ID} chargé sur {DEVICE}")
    return model


def _vjepa_inference(model, images, **kwargs):
    x = images.to(DEVICE).float()
    if x.max() > 1.5:
        x = x / 255.0
    x = x[..., :3].permute(0, 3, 1, 2)
    x = F.interpolate(x, size=(RES, RES), mode="bilinear", align_corners=False)
    x = x.unsqueeze(1).repeat(1, 2, 1, 1, 1)  # (N, T=2, 3, H, W)
    x = (x - _MEAN.to(x.device)) / _STD.to(x.device)
    with torch.no_grad():
        out = model(pixel_values_videos=x.to(torch.bfloat16), skip_predictor=True)
    return out.last_hidden_state.float().mean(dim=1)  # (N, 1024)


@configclass
class VJEPAObservationCfg:
    @configclass
    class VJEPAPolicyCfg(ObsGroup):
        image = ObsTerm(
            func=mdp.image_features,
            params={
                "sensor_cfg": SceneEntityCfg("tiled_camera"),
                "data_type": "rgb",
                "model_name": "vjepa2-vitl",
                "model_device": DEVICE,
                "model_zoo_cfg": {
                    "vjepa2-vitl": {"model": _load_vjepa, "inference": _vjepa_inference},
                },
            },
        )

    policy: ObsGroup = VJEPAPolicyCfg()


@configclass
class CartpoleVJEPACameraEnvCfg(CartpoleRGBCameraEnvCfg):
    observations: VJEPAObservationCfg = VJEPAObservationCfg()


_base = gym.spec("Isaac-Cartpole-RGB-ResNet18-v0")
gym.register(
    id="Isaac-Cartpole-RGB-VJEPA2-v0",
    entry_point=_base.entry_point,
    disable_env_checker=True,
    kwargs={**_base.kwargs, "env_cfg_entry_point": f"{__name__}:CartpoleVJEPACameraEnvCfg"},
)
