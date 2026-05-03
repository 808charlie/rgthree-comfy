import folder_paths
import comfy.lora
from comfy.utils import load_torch_file
import torch as th

from typing import Union

from .constants import get_category, get_name
from .power_prompt_utils import get_lora_by_filename
from .utils import FlexibleOptionalInputType, any_type
from .server.utils_info import get_model_info_file_data
from .log import log_node_warn

NODE_NAME = get_name('Power Ltx Lora Loader')


class RgthreePowerLtxLoraLoader:
  """ The Power Ltx Lora Loader is a powerful, flexible node to add multiple loras to a model/clip."""

  NAME = NODE_NAME
  CATEGORY = get_category()

  @classmethod
  def INPUT_TYPES(cls):  # pylint: disable = invalid-name, missing-function-docstring
    return {
      "required": {
      },
      # Since we will pass any number of loras in from the UI, this needs to always allow an
      "optional": FlexibleOptionalInputType(type=any_type, data={
        "model": ("MODEL",),
        "clip": ("CLIP",),
      }),
      "hidden": {},
    }

  RETURN_TYPES = ("MODEL", "CLIP")
  RETURN_NAMES = ("MODEL", "CLIP")
  FUNCTION = "load_loras"

  def load_loras(self, model=None, clip=None, **kwargs):
    """Loops over the provided loras in kwargs and applies valid ones with type multipliers."""
    for key, value in kwargs.items():
      key = key.upper()
      if key.startswith('LORA_') and 'on' in value and 'lora' in value and 'strength' in value:
        strength_model = value['strength']
        strength_clip = value['strengthTwo'] if 'strengthTwo' in value else None
        if clip is None:
          if strength_clip is not None and strength_clip != 0:
            log_node_warn(NODE_NAME, 'Recieved clip strength eventhough no clip supplied!')
          strength_clip = 0
        else:
          strength_clip = strength_clip if strength_clip is not None else strength_model
        if value['on'] and (strength_model != 0 or strength_clip != 0):
          lora = get_lora_by_filename(value['lora'], log_node=self.NAME)
          if lora is None:
            continue

          v2a = value.get('v2a') if 'v2a' in value else 1.0
          a2v = value.get('a2v') if 'a2v' in value else 1.0
          aud = value.get('aud') if 'aud' in value else 1.0
          vid = value.get('vid') if 'vid' in value else 1.0
          other = value.get('other') if 'other' in value else 1.0

          lora_path = folder_paths.get_full_path("loras", lora)
          if not lora_path:
            continue

          lora_data = load_torch_file(lora_path, safe_load=True)
          key_map = {}
          if model is not None:
            key_map = comfy.lora.model_lora_keys_unet(model.model, key_map)
          loaded = comfy.lora.load_lora(lora_data, key_map)

          keys_to_delete = []
          for k in list(loaded.keys()):
            k_str = k if isinstance(k, str) else (k[0] if isinstance(k, tuple) else str(k))
            multiplier = None
            if "video_to_audio_attn" in k_str:
              multiplier = v2a
            elif "audio_to_video_attn" in k_str:
              multiplier = a2v
            elif "audio_attn" in k_str or "audio_ff.net" in k_str:
              multiplier = aud
            elif "attn" in k_str or "ff.net" in k_str:
              multiplier = vid
            else:
              multiplier = other

            if multiplier == 0 or (isinstance(multiplier, float) and abs(float(multiplier)) < 1e-6):
              keys_to_delete.append(k)
            elif isinstance(multiplier, (float, int)) and abs(float(multiplier) - 1.0) > 1e-6:
              v = loaded[k]
              if hasattr(v, 'weights') and isinstance(getattr(v, 'weights'), tuple) and len(v.weights) >= 1:
                weights_list = list(v.weights)
                if th.is_tensor(weights_list[0]):
                  new_weights_tuple = (weights_list[0] * multiplier,) + tuple(weights_list[1:])
                  v.weights = new_weights_tuple

          for k in keys_to_delete:
            if k in loaded:
              del loaded[k]

          if model is not None:
            model = model.clone()
            model.add_patches(loaded, strength_model)
          if clip is not None and strength_clip != 0:
            clip = clip.clone()
            clip.add_patches(loaded, strength_clip)

    return (model, clip)

  @classmethod
  def get_enabled_loras_from_prompt_node(cls,
                                         prompt_node: dict) -> list[dict[str, Union[str, float]]]:
    """Gets enabled loras of a node within a server prompt."""
    result = []
    for name, lora in prompt_node['inputs'].items():
      if name.startswith('lora_') and lora['on']:
        lora_file = get_lora_by_filename(lora['lora'], log_node=cls.NAME)
        if lora_file is not None:  # Add the same safety check
          lora_dict = {
            'name': lora['lora'],
            'strength': lora['strength'],
            'path': folder_paths.get_full_path("loras", lora_file)
          }
          if 'strengthTwo' in lora:
            lora_dict['strength_clip'] = lora['strengthTwo']
          result.append(lora_dict)
    return result

  @classmethod
  def get_enabled_triggers_from_prompt_node(cls, prompt_node: dict, max_each: int = 1):
    """Gets trigger words up to the max for enabled loras of a node within a server prompt."""
    loras = [l['name'] for l in cls.get_enabled_loras_from_prompt_node(prompt_node)]
    trained_words = []
    for lora in loras:
      info = get_model_info_file_data(lora, 'loras', default={})
      if not info or not info.keys():
        log_node_warn(
          NODE_NAME,
          f'No info found for lora {lora} when grabbing triggers. Have you generated an info file'
          ' from the Power Ltx Lora Loader "Show Info" dialog?'
        )
        continue
      if 'trainedWords' not in info or not info['trainedWords']:
        log_node_warn(
          NODE_NAME,
          f'No trained words for lora {lora} when grabbing triggers. Have you fetched data from'
          'civitai or manually added words?'
        )
        continue
      trained_words += [w for wi in info['trainedWords'][:max_each] if (wi and (w := wi['word']))]
    return trained_words
