# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved
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

import unittest

import paddle
import paddle.distributed.fleet as fleet

from fleetx.data.tokenizers import GPTTokenizer
from fleetx.inference import InferenceEngine


class TestInferenceEngine(unittest.TestCase):
    def setUp(self):
        self.texts = [
            "Question: Who is the CEO of Apple? Answer:",
            "Question: Who is the CEO of Facebook? Answer:",
            "Question: How tall is the highest peak in the world? Answer:",
            "Question: Who is the president of the united states? Answer:",
            "Question: Where is the capital of France? Answer:",
            "Question: What is the largest animal in the ocean? Answer:",
            "Question: Who is the chancellor of Germany? Answer:",
        ]
        # TODO: change to download model
        self.model = './inference_model_pp1mp2'
        self.tokenizer_cls = 'gpt2'

    def test_main(self):
        fleet.init(is_collective=True)
        infer_engine = InferenceEngine(self.model)
        tokenizer = GPTTokenizer.from_pretrained(self.tokenizer_cls)

        inputs = tokenizer(
            text,
            padding=True,
            return_attention_mask=True,
            return_position_ids=True)
        ids = np.array(inputs["input_ids"]).reshape(len(text),
                                                    -1).astype('int64')
        attention_mask = np.array(inputs["attention_mask"]).reshape(
            len(text), -1).astype('float32')
        position_ids = np.array(inputs["position_ids"]).reshape(
            len(text), -1).astype('int64')

        data = [ids, attention_mask, position_ids]

        outs = infer_engine.predict(data)
        assert len(outs) == len(self.texts)
        for k, v in outs.items():
            for i in range(v.shape[0]):
                out_ids = [int(x) for x in v[i]]
                ret_str = tokenizer.convert_ids_to_string(out_ids)
                ret_str = text[i] + ret_str
                print(ret_str)


if __name__ == "__main__":
    unittest.main()
