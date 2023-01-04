_base_ = [
     '../_base_/models/upernet_swin_3band.py',
     '../_base_/datasets/reservoirs.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k_lowlr.py'
]
model = dict(
    decode_head=dict(
        num_classes=2,
        out_channels=2,
    loss_decode=dict(
        type='FocalLoss', use_sigmoid=True, loss_weight=2.0,
         )),
    auxiliary_head=dict(
        num_classes=2,
        out_channels=2,
    loss_decode=dict(
        type='FocalLoss', use_sigmoid=True, loss_weight=1.6,
         ))
    )
evaluation = dict(
    interval=1000,  # The interval of evaluation.
    metric='mIoU')  # The evaluation metric.
checkpoint_config = dict(  # Config to set the checkpoint hook, Refer to https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/checkpoint.py for implementation.
    by_epoch=False,  # Whether count by epoch or not.
    interval=1000)  # The save interval.
