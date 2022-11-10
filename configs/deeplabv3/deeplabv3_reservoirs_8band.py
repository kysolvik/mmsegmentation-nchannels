_base_ = [
    '../_base_/models/deeplabv3_r50-d8.py',
    '../_base_/datasets/reservoirs_8band.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]
model = dict(
    decode_head=dict(
        num_classes=2,
        out_channels=2,
    loss_decode=dict(
        type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0, class_weight=[0.5, 442.],
         )),
    auxiliary_head=dict(
        num_classes=2,
        out_channels=2,
    loss_decode=dict(
        type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0, class_weight=[0.5, 442.],
         ))
    )
evaluation = dict(
    interval=1000,  # The interval of evaluation.
    metric='mIoU')  # The evaluation metric.
checkpoint_config = dict(  # Config to set the checkpoint hook, Refer to https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/checkpoint.py for implementation.
    by_epoch=False,  # Whether count by epoch or not.
    interval=1000)  # The save interval.

