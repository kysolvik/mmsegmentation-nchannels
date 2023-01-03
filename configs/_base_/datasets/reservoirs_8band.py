# dataset settings
dataset_type = 'CustomDataset'
data_root = 'data/reservoirs_8band/'
img_norm_cfg = dict(
    mean=[125.09924004, 162.62489442, 130.07100181,  99.07333485,
          65.60963343, 141.6956218 ,  91.84874383, 180.89670207],
    std=[27.843059  , 35.68728997, 30.97197954, 44.67461069, 43.33962304,
         29.72645861, 21.93850306, 31.91190727], to_rgb=False)
crop_size = (256, 256)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='Resize', img_scale=(512, 512), ratio_range=(1.0, 1.5)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=1.0),
    dict(type='RandomFlip', prob=0.5),
#     dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
#     dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=1),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512,512),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
#             dict(type='Resize', img_scale=(512,512), ratio_range=(1.0, 1.5), keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
     samples_per_gpu=4,
     workers_per_gpu=4,
     train=dict(
        type=dataset_type,
        img_suffix='.tif',
        seg_map_suffix='.png',
        classes=['BG', 'Water'],
        data_root=data_root,
        img_dir='img_dir/train',
        ann_dir='ann_dir/train',
        pipeline=train_pipeline,
        palette=[[0,0,0],[255,255,255]]),
     val=dict(
        type=dataset_type,
        img_suffix='.tif',
        seg_map_suffix='.png',
        classes=['BG', 'Water'],
        data_root=data_root,
        img_dir='img_dir/val',
        ann_dir='ann_dir/val',
        pipeline=test_pipeline,
        palette=[[0,0,0],[255,255,255]]),
     test=dict(
        type=dataset_type,
        img_suffix='.tif',
        seg_map_suffix='.png',
        classes=['BG', 'Water'],
        data_root=data_root,
        img_dir='img_dir/val',
        ann_dir='ann_dir/val',
        pipeline=test_pipeline,
        palette=[[0,0,0],[255,255,255]]),
     )

