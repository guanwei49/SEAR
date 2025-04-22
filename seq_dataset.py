import os
from copy import deepcopy

import numpy as np
from recbole.data.dataset import SequentialDataset

import torch

from recbole.data.interaction import Interaction
from recbole.utils import FeatureType,set_color

class CustomizedSeqDataset(SequentialDataset):
    def __init__(self, config):
        super().__init__(config)

    def _fill_nan(self):
        """Missing value imputation.

        For fields with type :obj:`~recbole.utils.enum_type.FeatureType.TOKEN`, missing value will be filled by
        ``[PAD]``, which indexed as 0.

        For fields with type :obj:`~recbole.utils.enum_type.FeatureType.FLOAT`, missing value will be filled by
        the average of original data.
        """
        self.logger.debug(set_color("Filling nan", "green"))

        for feat_name in self.feat_name_list:
            feat = getattr(self, feat_name)
            for field in feat:
                ftype = self.field2type[field]
                if ftype == FeatureType.TOKEN:
                    feat[field].fillna(value=0, inplace=True)
                elif ftype == FeatureType.FLOAT:
                    feat[field].fillna(value=np.inf, inplace=True)
                else:
                    dtype = np.int64 if ftype == FeatureType.TOKEN_SEQ else np.float
                    feat[field] = feat[field].apply(
                        lambda x: np.array([], dtype=dtype)
                        if isinstance(x, float)
                        else x
                    )

    def data_augmentation(self):
        """Augmentation processing for sequential dataset.

        E.g., ``u1`` has purchase sequence ``<i1, i2, i3, i4>``,
        then after augmentation, we will generate three cases.

        ``u1, <i1> | i2``

        (Which means given user_id ``u1`` and item_seq ``<i1>``,
        we need to predict the next item ``i2``.)

        The other cases are below:

        ``u1, <i1, i2> | i3``

        ``u1, <i1, i2, i3> | i4``
        """
        self.logger.debug("data_augmentation")

        self._aug_presets()

        self._check_field("uid_field", "time_field")
        max_item_list_len = self.config["MAX_ITEM_LIST_LENGTH"]
        self.sort(by=[self.uid_field, self.time_field], ascending=True)
        last_uid = None
        uid_list, item_list_index, target_index,  target_list_index, item_list_length = [], [], [], [], []
        seq_start = 0

        values, counts = torch.unique(self.inter_feat[self.uid_field], return_counts=True)
        uid_count_dict = dict(zip(values.tolist(), counts.tolist()))
        for i, uid in enumerate(self.inter_feat[self.uid_field].numpy()):
            if last_uid != uid:
                last_uid = uid
                seq_start = i
            else:
                if i - seq_start > max_item_list_len:
                    seq_start += 1
                uid_list.append(uid)
                item_list_index.append(slice(seq_start, i))
                target_index.append(i)
                target_list_index.append(slice(i, min(i + self.config['k'] , seq_start+uid_count_dict[uid] -2))) # -2 is for avoid exposing the eval and test data.
                item_list_length.append(i - seq_start)

        item_list_index = np.array(item_list_index)
        target_index = np.array(target_index)
        target_list_index = np.array(target_list_index)
        item_list_length = np.array(item_list_length, dtype=np.int64)


        new_length = len(item_list_index)
        new_data = self.inter_feat[target_index]
        new_dict = {
            self.item_list_length_field: torch.tensor(item_list_length),
        }

        ## for target_list_index
        new_dict['target_id_list'] = torch.zeros(
            (len(target_list_index), self.config['k']), dtype=int
        )
        value = self.inter_feat[self.iid_field]
        for i, index in enumerate(target_list_index):
            item_temp =  value[index]
            new_dict['target_id_list'][i][:len(item_temp)] = item_temp


        for field in self.inter_feat:
            if field != self.uid_field:
                list_field = getattr(self, f"{field}_list_field")
                list_len = self.field2seqlen[list_field]
                shape = (
                    (new_length, list_len)
                    if isinstance(list_len, int)
                    else (new_length,) + list_len
                )
                if (
                        self.field2type[field] in [FeatureType.FLOAT, FeatureType.FLOAT_SEQ]
                        and field in self.config["numerical_features"]
                ):
                    shape += (2,)
                new_dict[list_field] = torch.zeros(
                    shape, dtype=self.inter_feat[field].dtype
                )

                value = self.inter_feat[field]
                for i, (index, length) in enumerate(
                        zip(item_list_index, item_list_length)
                ):
                    new_dict[list_field][i][:length] = value[index]

        new_data.update(Interaction(new_dict))
        self.inter_feat = new_data

