CUDA_VISIBLE_DEVICES=0 python main_no_retrain.py \
--dataset cifar100 \
--arch resnet \
--depth 101 \
--lr 0.001 \
--epochs 160 \
--schedule 80 120 \
--batch-size 256 \
--test-batch-size 128 \
--save ./baseline/resnet101unpruned-cifar100/pruned_0.3/retrain_0.3 \
--momentum 0.9 \
--sparsity-regularization \
--scratch ./baseline/resnet101unpruned-cifar100/pruned_0.3/pruned.pth.tar \
--start-epoch 0

CUDA_VISIBLE_DEVICES=0 python main_no_retrain.py \
--dataset cifar100 \
--arch resnet \
--depth 101 \
--lr 0.0001 \
--epochs 160 \
--schedule 80 120 \
--batch-size 256 \
--test-batch-size 128 \
--save ./baseline/resnet101unpruned-cifar100/pruned_0.3/retrain_0.3 \
--momentum 0.9 \
--sparsity-regularization \
--scratch ./baseline/resnet101unpruned-cifar100/pruned_0.3/pruned.pth.tar \
--start-epoch 0

CUDA_VISIBLE_DEVICES=0 python main_no_retrain.py \
--dataset cifar100 \
--arch resnet \
--depth 101 \
--lr 0.01 \
--epochs 160 \
--schedule 80 120 \
--batch-size 256 \
--test-batch-size 128 \
--save ./baseline/resnet101unpruned-cifar100/pruned_0.3/retrain_0.3 \
--momentum 0.9 \
--sparsity-regularization \
--scratch ./baseline/resnet101unpruned-cifar100/pruned_0.3/pruned.pth.tar \
--start-epoch 0
