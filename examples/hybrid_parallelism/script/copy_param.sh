
out=ernie_xxlarge_untie48
rm -rf ${out}
mkdir ${out}
cp last_five_step_merge/* ${out}

for p in _multi_head_att_key_fc.b_0 _multi_head_att_value_fc.b_0 _multi_head_att_key_fc.w_0 _multi_head_att_value_fc.w_0 _ffn_fc_0.b_0  _multi_head_att_output_fc.b_0  _post_att_layer_norm_bias _ffn_fc_0.w_0  _multi_head_att_output_fc.w_0 _post_att_layer_norm_scale _ffn_fc_1.b_0 _multi_head_att_query_fc.b_0 _post_ffn_layer_norm_bias _ffn_fc_1.w_0 _multi_head_att_query_fc.w_0 _post_ffn_layer_norm_scale;do
    for i in `seq 1 23`;do
        cp last_five_step_merge/encoder_layer_0${p} ${out}/encoder_layer_${i}${p}
    done
    for i in `seq 24 47`;do
        cp last_five_step_merge/encoder_layer_1${p} ${out}/encoder_layer_${i}${p}
    done
done
