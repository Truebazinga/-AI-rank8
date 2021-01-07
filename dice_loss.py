# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..builder import LOSSES
from .utils import weight_reduce_loss


@LOSSES.register_module()
class DiceLoss(nn.Module):
    """
    Implements the dice loss function.
    Args:
        ignore_index (int64): Specifies a target value that is ignored
            and does not contribute to the input gradient. Default ``255``.
    """

    def __init__(self, ignore_index=255):
        super(DiceLoss, self).__init__()
        self.ignore_index = ignore_index
        self.eps = 1e-5

    def forward(self,cls_score,label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        if len(label.shape) != len(cls_score.shape):
            label = torch.unsqueeze(label, 1)
        num_classes = cls_score.shape[1]
        mask = (label != self.ignore_index)
        cls_score = cls_score * mask

        label = label.int()
        # label = np.array(label.cpu(),dtype=np.int32)
        # label = torch.from_numpy(label)

        #label = torch.can_cast(label, dtype='int32')
        single_label_lists = []
        for c in range(num_classes):
            #single_label = torch.can_cast((label == c), dtype='int32')
            single_label = (label == c).int()
            # single_label = np.array(label.cpu()==c,dtype=np.int32)
            # single_label = torch.from_numpy(single_label)

            single_label = torch.squeeze(single_label, axis=1)
            single_label_lists.append(single_label)
        label_one_hot = torch.stack(tuple(single_label_lists), axis=1)
        cls_score = F.softmax(cls_score, dim=1)

        label_one_hot = label_one_hot.float()
        # label_one_hot = np.array(label_one_hot.cpu(),dtype=np.float32)
        # label_one_hot = torch.from_numpy(label_one_hot)

        #label_one_hot = orch.can_cast(label_one_hot, dtype='float32')
        dims = (0,) + tuple(range(2, label.ndimension()))
        intersection = torch.sum(cls_score * label_one_hot, dims)
        cardinality = torch.sum(cls_score + label_one_hot, dims)
        dice_loss = (2. * intersection / (cardinality + self.eps)).mean()

        return 1 - dice_loss