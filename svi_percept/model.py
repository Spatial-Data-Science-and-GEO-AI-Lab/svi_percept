#  svi_percept - KNN model
#  Copyright (C) 2024 Matthew Danish
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

from transformers import PreTrainedModel, PretrainedConfig
from sklearn.utils.extmath import softmax
import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download


class SVIPerceptConfig(PretrainedConfig):
    model_type = "svi_percept"

    def __init__(self,
                repo_id: str = "Spatial-Data-Science-and-GEO-AI-Lab/svi_percept",  # Default repo
                **kwargs):
        self.repo_id = repo_id
        self.categories = ['walkability', 'bikeability', 'pleasantness', 'greenness', 'safety']
        self.k = 40
        super().__init__(**kwargs)

class SVIPerceptModel(PreTrainedModel):
    config_class = SVIPerceptConfig

    def __init__(self, config, repo_id=None, categories=None, k=None):
        super().__init__(config)
        self.repo_id = repo_id or config.repo_id
        self.categories = categories or config.categories
        self.k = k or config.k
        # Load npz data from model repository
        npz_path = hf_hub_download(
            repo_id=self.repo_id,
            filename="data.npz",
            repo_type="model"
        )
        npz_data = np.load(npz_path)
        self.data = npz_data

        self.matrices = nn.ParameterDict({
            cat: nn.Parameter(
                torch.from_numpy(npz_data[f'{cat}_vecs']).to(torch.float32),
                requires_grad=False
            ) for cat in self.categories
        })

        self.scores = nn.ParameterDict({
            cat: nn.Parameter(
                torch.from_numpy(npz_data[f'{cat}_scores']).to(torch.float32),
                requires_grad=False
            ) for cat in self.categories
        })

    def forward(self, features, **kwargs):
        results = np.zeros((features.shape[0], len(self.categories)))

        device = features.device
        for cat_i, cat in enumerate(self.categories):
            allvecs = self.matrices[cat].to(device)
            scores = self.scores[cat].to(device)
            similarities = torch.matmul(features, torch.transpose(allvecs, 0, 1))  # [batch_size, num_vectors]
            raw_weights, indices = torch.topk(similarities, k=self.k, dim=1)  # [batch_size, self.k]
            kweights = torch.pow(10.0, raw_weights)  # [batch_size, self.k]
            kweights = torch.nn.functional.softmax(kweights, dim=1)  # [batch_size, self.k]
            kscores = scores[indices]  # [batch_size, self.k]
            results[:, cat_i] = torch.sum(kscores * kweights, dim=1).cpu()
        return {"results": results}
