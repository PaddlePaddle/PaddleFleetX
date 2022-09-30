
import sys

sys.path.append("/work/t5_project/FleetX/")

import random
import numpy as np
import paddle
from ppfleetx.models.multimodal_model.imagen import imagen_397M_text2im_64
from transformers import T5ForConditionalGeneration, T5Tokenizer

from paddle.jit import to_static
from paddle.static import InputSpec


if __name__ == "__main__":
    paddle.seed(100)
    np.random.seed(100)
    random.seed(100)

    model_config = {
        "use_t5": True,
        "return_all_unet_outputs": True,
    }
    # model_config = {'text_encoder_name': '/work/t5_project/imagen/bak/imagen-infer/t5/t5-small', 'timesteps': 270, 'in_chans': 3, 'noise_schedules': 'cosine', 'pred_objectives': 'noise', 'lowres_noise_schedule': 'linear', 'lowres_sample_noise_level': 0.2, 'per_sample_random_aug_noise_level': False, 'condition_on_text': True, 'auto_normalize_img': True, 'p2_loss_weight_gamma': 0.5, 'p2_loss_weight_k': 1.0, 'dynamic_thresholding': True, 'dynamic_thresholding_percentile': 0.95, 'only_train_unet_number': None}
    imagen_model = imagen_397M_text2im_64(**model_config)
    imagen_model.eval()
    # input_texts = ["The weather is sunny today, the scenery is beautiful everywhere"]
    # tokenizer = T5Tokenizer.from_pretrained("/work/t5_project/t5/t5-3b/")
    # input_token = tokenizer(input_texts, padding=True)

    x_spec = InputSpec(shape=[1, None], dtype='int64', name='input_ids')
    y_spec = InputSpec(shape=[1, None], dtype='int64', name='text_masks')


    input_ids = 128 * np.ones((1, 128)).astype(np.int64)
    text_masks = np.ones((1, 128)).astype(np.int64)

    # input_ids = np.array(input_token.input_ids).reshape((1, -1)).astype(np.int64)
    # text_masks = np.ones(input_ids.shape).astype(np.int64)
    print("input_ids shape: ", input_ids.shape, "text_masks shape: ", text_masks.shape)

    out = imagen_model.sample(paddle.to_tensor(input_ids), paddle.to_tensor(text_masks))
    # out = sample(input_ids, text_masks)
    print("eval: ", out[0].numpy().reshape(-1).tolist()[:20])

    sample_static = to_static(imagen_model.sample, [x_spec, y_spec])
    for block in sample_static.main_program.blocks:
        for op in block.ops:
            if op.has_attr("op_callstack"):
                op._remove_attr("op_callstack")
    # paddle.seed(100)
    # np.random.seed(100)
    # random.seed(100)
    out2 = sample_static(input_ids, text_masks)
    print("to static: ", out2[0].numpy().reshape(-1).tolist()[:20])

    paddle.jit.save(sample_static, path=f'/work/t5_project/imagen/models/imagen_v1')