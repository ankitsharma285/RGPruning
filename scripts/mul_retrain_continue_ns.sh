CUDA_VISIBLE_DEVICES=0 python main_c_vgg.py \
--dataset cifar100 \
--arch vgg \
--depth 16 \
--lr 0.001 \
--epochs 40 \
--schedule 80 120 \
--batch-size 256 \
--test-batch-size 128 \
--save ./baseline/unpruned-vgg16-cifar100/retrain_0.1 \
--momentum 0.9 \
--sparsity-regularization \
--scratch ./baseline/unpruned-vgg16-cifar100/pruned_0.1/pruned.pth.tar \
--start-epoch 0 \
--seed 1

CUDA_VISIBLE_DEVICES=0 python main_c_vgg.py \
--dataset cifar100 \
--arch vgg \
--depth 16 \
--lr 0.001 \
--epochs 40 \
--schedule 80 120 \
--batch-size 256 \
--test-batch-size 128 \
--save ./baseline/unpruned-vgg16-cifar100/retrain_0.2 \
--momentum 0.9 \
--sparsity-regularization \
--scratch ./baseline/unpruned-vgg16-cifar100/pruned_0.2/pruned.pth.tar \
--start-epoch 0 \
--seed 1

CUDA_VISIBLE_DEVICES=0 python main_c_vgg.py \
--dataset cifar100 \
--arch vgg \
--depth 16 \
--lr 0.001 \
--epochs 40 \
--schedule 80 120 \
--batch-size 256 \
--test-batch-size 128 \
--save ./baseline/unpruned-vgg16-cifar100/retrain_0.3 \
--momentum 0.9 \
--sparsity-regularization \
--scratch ./baseline/unpruned-vgg16-cifar100/pruned_0.3/pruned.pth.tar \
--start-epoch 0 \
--seed 1

CUDA_VISIBLE_DEVICES=0 python main_c_vgg.py \
--dataset cifar100 \
--arch vgg \
--depth 16 \
--lr 0.001 \
--epochs 40 \
--schedule 80 120 \
--batch-size 256 \
--test-batch-size 128 \
--save ./baseline/unpruned-vgg16-cifar100/retrain_0.4 \
--momentum 0.9 \
--sparsity-regularization \
--scratch ./baseline/unpruned-vgg16-cifar100/pruned_0.4/pruned.pth.tar \
--start-epoch 0 \
--seed 1

CUDA_VISIBLE_DEVICES=0 python main_c_vgg.py \
--dataset cifar100 \
--arch vgg \
--depth 16 \
--lr 0.001 \
--epochs 40 \
--schedule 80 120 \
--batch-size 256 \
--test-batch-size 128 \
--save ./baseline/unpruned-vgg16-cifar100/retrain_0.5 \
--momentum 0.9 \
--sparsity-regularization \
--scratch ./baseline/unpruned-vgg16-cifar100/pruned_0.5/pruned.pth.tar \
--start-epoch 0 \
--seed 1

CUDA_VISIBLE_DEVICES=0 python main_c_vgg.py \
--dataset cifar100 \
--arch vgg \
--depth 16 \
--lr 0.001 \
--epochs 40 \
--schedule 80 120 \
--batch-size 256 \
--test-batch-size 128 \
--save ./baseline/unpruned-vgg16-cifar100/retrain_0.6 \
--momentum 0.9 \
--sparsity-regularization \
--scratch ./baseline/unpruned-vgg16-cifar100/pruned_0.6/pruned.pth.tar \
--start-epoch 0 \
--seed 1

CUDA_VISIBLE_DEVICES=0 python main_c_vgg.py \
--dataset cifar100 \
--arch vgg \
--depth 16 \
--lr 0.001 \
--epochs 40 \
--schedule 80 120 \
--batch-size 256 \
--test-batch-size 128 \
--save ./baseline/unpruned-vgg16-cifar100/retrain_0.7 \
--momentum 0.9 \
--sparsity-regularization \
--scratch ./baseline/unpruned-vgg16-cifar100/pruned_0.7/pruned.pth.tar \
--start-epoch 0 \
--seed 1

CUDA_VISIBLE_DEVICES=0 python main_c_vgg.py \
--dataset cifar100 \
--arch vgg \
--depth 16 \
--lr 0.001 \
--epochs 40 \
--schedule 80 120 \
--batch-size 256 \
--test-batch-size 128 \
--save ./baseline/unpruned-vgg16-cifar100/retrain_0.8 \
--momentum 0.9 \
--sparsity-regularization \
--scratch ./baseline/unpruned-vgg16-cifar100/pruned_0.8/pruned.pth.tar \
--start-epoch 0 \
--seed 1

CUDA_VISIBLE_DEVICES=0 python main_c_vgg.py \
--dataset cifar100 \
--arch vgg \
--depth 16 \
--lr 0.001 \
--epochs 40 \
--schedule 80 120 \
--batch-size 256 \
--test-batch-size 128 \
--save ./baseline/unpruned-vgg16-cifar100/retrain_0.9 \
--momentum 0.9 \
--sparsity-regularization \
--scratch ./baseline/unpruned-vgg16-cifar100/pruned_0.9/pruned.pth.tar \
--start-epoch 0 \
--seed 1
