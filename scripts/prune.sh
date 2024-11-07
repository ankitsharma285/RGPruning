python mVggprune.py \
--dataset cifar100 \
--test-batch-size 256 \
--depth 16 \
--percent 0.3 \
--model ./baseline/vgg-cifar100/main_run/best_model.pth.tar \
--save ./baseline/vgg-cifar100/pruned_0.3 \
#--gpu_ids 0

python mVggprune.py \
--dataset cifar100 \
--test-batch-size 256 \
--depth 16 \
--percent 0.4 \
--model ./baseline/vgg-cifar100/main_run/best_model.pth.tar \
--save ./baseline/vgg-cifar100/pruned_0.4 \
