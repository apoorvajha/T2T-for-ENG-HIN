# coding=utf-8
# Copyright 2018 The Tensor2Tensor Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Data generators for translation data-sets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tarfile

# Depenhincy imports

from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import translate
from tensor2tensor.utils import registry

import tensorflow as tf

FLAGS = tf.flags.FLAGS

# End-of-sentence marker.
EOS = text_encoder.EOS_ID


_enhi_TRAIN_DATASETS = [
    [
        "~/t2t/train_enhi.gz",  # pylint: disable=line-too-long
        ("training/train.en",
         "training/train.hi")
    ],
]

_enhi_TEST_DATASETS = [
    [
        "~/t2t/dev.gz",
        ("dev/dev.en", "dev/dev.hi")
    ],
]


@registry.register_problem
class Translateenhi(translate.TranslateProblem):
  """Problem spec for WMT En-hi translation."""

  @property
  def targeted_vocab_size(self):
    return 2**13  # 8192

  @property
  def vocab_name(self):
    return "vocab.enhi"

  @property
  def vocab_filename(self):
    return "vocab.enhi.%d" % self.approx_vocab_size
  
  def source_data_files(self, dataset_split):
    train = dataset_split == problem.DatasetSplit.TRAIN
    return _enhi_TRAIN_DATASETS if train else _enhi_TEST_DATASETS

  def vocab_data_files(self):
    datasets = self.source_data_files(problem.DatasetSplit.TRAIN)
    vocab_datasets = []
    if datasets[0][0].endswith("data-plaintext-format.tar"):
      vocab_datasets.append([
          datasets[0][0], [
              "%s-compiled-train.lang1" % self.name,
              "%s-compiled-train.lang2" % self.name
          ]
      ])
      datasets = datasets[1:]
    vocab_datasets += [[item[0], [item[1][0], item[1][1]]] for item in datasets]
    return vocab_datasets


  def generator(self, data_dir, tmp_dir, train):
    symbolizer_vocab = generator_utils.get_or_generate_vocab(
        data_dir, tmp_dir, self.vocab_file, self.targeted_vocab_size,
        _enhi_TRAIN_DATASETS)
    datasets = _enhi_TRAIN_DATASETS if train else _enhi_TEST_DATASETS
    tag = "train" if train else "dev"
    data_path = translate.compile_data(tmp_dir, datasets,
                                       "enhi_tok_%s" % tag)
    return translate.token_generator(data_path + ".lang1", data_path + ".lang2",
                                     symbolizer_vocab, EOS)

  @property
  def input_space_id(self):
    return problem.SpaceID.EN_TOK

  @property
  def target_space_id(self):
    return problem.SpaceID.EN_TOK
    #return problem.SpaceID.HI_TOK


@registry.register_problem
class Translateenhi_main(Translateenhi):

  @property
  def targeted_vocab_size(self):
    return 2**15  # 32768




