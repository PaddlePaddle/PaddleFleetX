import os

not_rst_file = [
    'fleet_and_edl_for_distillation_cn.md',
    'fleet_and_edl_for_distillation.md',
    'fleet_collective_training_practices_cn.md',
    'fleet_collective_training_practices.md',
    'fleet_collective_training_speedup_with_amp_cn.md',
    'fleet_collective_training_speedup_with_amp.md',
    'fleet_improve_large_batch_accuracy_cn.md',
    'fleet_improve_large_batch_accuracy.md',
    'fleet_on_cloud_cn.md',
    'fleet_on_cloud.md',
    'fleet_dataset_w2v_cn.md',
    ]

files = os.listdir(".")
files = [file for file in files if file not in not_rst_file]
print(files)
for f in files:
    if "_cn" not in f:
        continue
    if ".md" in f:
        os.system("pandoc {} -f markdown -t rst -o ../paddle_fleet_rst/{}".format(f, f.replace(".md", ".rst")))


