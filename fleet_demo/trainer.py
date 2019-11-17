from fleet.dataset import QueueDataset, MemoryDataset
from fleet.models import MultiSlotCTR
from fleet.trainer import OnlineTrainer
from fleet.optimizer import SGD

# step1: define the model, we prepare lots of predefined models
slot_filename = "slot.txt"
model = MultiSlotCTR()
model.set_data_generator_file("asq_reader.py") # default is reader.py
model.build_train_net(slot_filename=slot_filename)

# step2: define the dataset, and the reader
#dataset = QueueDataset()
dataset = MemoryDataset()

# step3: define the trainer
trainer = OnlineTrainer()
trainer.set_thread(10)
trainer.set_batch_size(32)
trainer.init(dataset=dataset, model=model, optimizer=SGD(0.0001))

# step4: do online training
train_folders = ["2100"]

for i, pass_folder in enumerate(train_folders):
    trainer.train_pass(pass_folder, prefix="part", is_debug=True)
    trainer.save_inference_model("{}_checkpoint".format(train_folders[i]))
