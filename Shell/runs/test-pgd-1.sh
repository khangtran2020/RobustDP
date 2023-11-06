CUDA_VISIBLE_DEVICES=5 python main.py --proj_name test-pgd \
        --gen_mode clean \
        --device gpu \
        --debug 1 \
        --data mnist \
        --lr 0.001 \
        --epochs 100 \
        --clipw 1.0 \
        --att_mode pgd-clean \
        --pgd_steps 50