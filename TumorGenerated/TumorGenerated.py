import random
from typing import Hashable, Mapping, Dict
import os
import glob
from monai.config import KeysCollection
from monai.config.type_definitions import NdarrayOrTensor
from monai.transforms.transform import MapTransform, RandomizableTransform

from .utils import SynthesisTumor, get_predefined_texture
import numpy as np

class TumorGenerated(RandomizableTransform, MapTransform):
    def __init__(self, 
    keys: KeysCollection, 
    prob: float = 0.1,
    tumor_prob = [0.2, 0.2, 0.2, 0.2, 0.2],
    allow_missing_keys: bool = False
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)
        #random.seed(0)
        #np.random.seed(0)

        self.tumor_types = ['tiny', 'small', 'medium', 'large', 'mix']
        
        assert len(tumor_prob) == 5
        self.tumor_prob = np.array(tumor_prob)
        # texture shape: 420, 300, 320
        # self.textures = pre_define 10 texture
        self.textures = []
        sigma_as = [3, 6, 9, 12, 15]
        sigma_bs = [4, 7]
        predefined_texture_shape = (420, 300, 320)
        for sigma_a in sigma_as:
            for sigma_b in sigma_bs:
                texture = get_predefined_texture(predefined_texture_shape, sigma_a, sigma_b)
                self.textures.append(texture)
        print("All predefined texture have generated.")

    def _get_next_case_id(self, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        existing = glob.glob(os.path.join(out_dir, "case_*.png"))
        numbers = []
        for f in existing:
            basename = os.path.basename(f)
            if basename.startswith("case_"):
                num_part = basename.split("_")[1]
                if num_part.isdigit():
                    numbers.append(int(num_part))
        next_num = (max(numbers) + 1) if numbers else 1
        return f"case_{next_num:03d}"
    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        self.randomize(None)

        if self._do_transform and (np.max(d['label']) <= 1):  
            tumor_type = np.random.choice(self.tumor_types, p=self.tumor_prob.ravel())
            texture = get_predefined_texture((420, 300, 320),
                                 np.random.choice([3,6,9,12,15]),
                                 np.random.choice([4,7]))
            print(f'Tumor type selection {tumor_type}')
            out_dir = "./png_steps/"
            case_id = self._get_next_case_id(out_dir)
            d['image'][0], d['label'][0] = SynthesisTumor(
            d['image'][0], 
            d['label'][0], 
            "small", 
            texture, 
            save_dir=out_dir, 
            case_id=case_id)
            # print(tumor_type, d['image'].shape, np.max(d['label']))
        return d
