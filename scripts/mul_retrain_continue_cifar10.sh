CUDA_VISIBLE_DEVICES=0 python main_c_vgg.py \
--dataset cifar10 \
--arch vgg \
--depth 16 \
--lr 0.1 \
--epochs 160 \
--schedule 80 120 \
--batch-size 256 \
--test-batch-size 128 \
--save ./baseline/lrS03-vgg16-cifar10/retrain_0.3 \
--momentum 0.9 \
--sparsity-regularization \
--scratch ./baseline/lrS03-vgg16-cifar10/pruned_0.3/pruned.pth.tar \
--start-epoch 0 \
--seed 1

CUDA_VISIBLE_DEVICES=0 python main_c_vgg.py \
--dataset cifar10 \
--arch vgg \
--depth 16 \
--lr 0.1 \
--epochs 160 \
--schedule 80 120 \
--batch-size 256 \
--test-batch-size 128 \
--save ./baseline/lrS03-vgg16-cifar10/retrain_0.4 \
--momentum 0.9 \
--sparsity-regularization \
--scratch ./baseline/lrS03-vgg16-cifar10/pruned_0.4/pruned.pth.tar \
--start-epoch 0 \
--seed 1

CUDA_VISIBLE_DEVICES=0 python main_c_vgg.py \
--dataset cifar100 \
--arch vgg \
--depth 16 \
--lr 0.1 \
--epochs 160 \
--schedule 80 120 \
--batch-size 256 \
--test-batch-size 128 \
--save ./baseline/lrS03-vgg16-cifar100/retrain_0.3 \
--momentum 0.9 \
--sparsity-regularization \
--scratch ./baseline/lrS03-vgg16-cifar100/pruned_0.3/pruned.pth.tar \
--start-epoch 0 \
--seed 1

CUDA_VISIBLE_DEVICES=0 python main_c_vgg.py \
--dataset cifar100 \
--arch vgg \
--depth 16 \
--lr 0.1 \
--epochs 160 \
--schedule 80 120 \
--batch-size 256 \
--test-batch-size 128 \
--save ./baseline/lrS03-vgg16-cifar100/retrain_0.4 \
--momentum 0.9 \
--sparsity-regularization \
--scratch ./baseline/lrS03-vgg16-cifar100/pruned_0.4/pruned.pth.tar \
--start-epoch 0 \
--seed 1
