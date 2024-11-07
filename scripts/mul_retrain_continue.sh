CUDA_VISIBLE_DEVICES=0 python main_c_resnet.py \
--dataset cifar100 \
--arch densenet \
--depth 40 \
--lr 0.1 \
--epochs 160 \
--schedule 80 120 \
--batch-size 256 \
--test-batch-size 128 \
--save ./baseline/lrS03-densenet-cifar100/retrain_0.3 \
--momentum 0.9 \
--sparsity-regularization \
--scratch ./baseline/lrS03-densenet-cifar100/pruned_0.3/pruned.pth.tar \
--start-epoch 0 \
--seed 1

CUDA_VISIBLE_DEVICES=0 python main_c_resnet.py \
--dataset cifar100 \
--arch densenet \
--depth 40 \
--lr 0.1 \
--epochs 160 \
--schedule 80 120 \
--batch-size 256 \
--test-batch-size 128 \
--save ./baseline/lrS03-densenet-cifar100/retrain_0.4 \
--momentum 0.9 \
--sparsity-regularization \
--scratch ./baseline/lrS03-densenet-cifar100/pruned_0.4/pruned.pth.tar \
--start-epoch 0 \
--seed 1

CUDA_VISIBLE_DEVICES=0 python mMainWT.py \
--dataset cifar100 \
--arch densenet \
--depth 40 \
--lr 0.1 \
--epochs 160 \
--schedule 80 160 \
--batch-size 256 \
--test-batch-size 128 \
--save ./baseline/unpruned-densenet-cifar100/main_run \
--momentum 0.9 \
--sparsity-regularization \
--seed 1
