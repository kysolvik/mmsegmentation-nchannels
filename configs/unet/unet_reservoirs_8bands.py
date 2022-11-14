_base_ = [
    '../_base_/models/pspnet_unet_s5-d16.py',
    '../_base_/datasets/reservoirs_8band.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]
model = dict(
    test_cfg=dict(crop_size=(512, 512), stride=(0, 0)),
    decode_head=dict(
        num_classes=2,
        out_channels=2,
    loss_decode=dict(
        type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0, class_weight=[0.5, 442.], #100 was pretty good
         )),
    auxiliary_head=dict(
        num_classes=2,
        out_channels=2,
    loss_decode=dict(
        type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0, class_weight=[0.5, 442.], # 100 seemed to work pretty well
         ))
    )
evaluation = dict(
    interval=1000,  # The interval of evaluation.
    metric='mIoU')  # The evaluation metric.
checkpoint_config = dict(  # Config to set the checkpoint hook, Refer to https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/checkpoint.py for implementation.
    by_epoch=False,  # Whether count by epoch or not.
    interval=1000)  # The save interval.

