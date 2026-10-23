ROOT = 'C:/Users/Dala/Documents/New project/mmaction_rebuild/configs/..'
default_hooks = dict(
    checkpoint=dict(
        interval=1,
        max_keep_ckpts=2,
        rule='greater',
        save_best='acc/top1',
        save_optimizer=True,
        type='CheckpointHook'),
    logger=dict(interval=1, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    runtime_info=dict(type='RuntimeInfoHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'))
default_scope = 'mmaction'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='gloo'),
    mp_cfg=dict(mp_start_method='spawn', opencv_num_threads=0))
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=1)
model = dict(
    backbone=dict(
        arch='tiny',
        drop_path_rate=0.1,
        patch_size=(
            2,
            4,
            4,
        ),
        pretrained=None,
        pretrained2d=False,
        type='SwinTransformer3D',
        window_size=(
            8,
            7,
            7,
        ),
        with_cp=True),
    cls_head=dict(
        average_clips='prob',
        dropout_ratio=0.5,
        in_channels=768,
        num_classes=10,
        spatial_type='avg',
        type='I3DHead'),
    data_preprocessor=dict(
        format_shape='NCTHW',
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='ActionDataPreprocessor'),
    type='Recognizer3D')
optim_wrapper = dict(
    clip_grad=dict(max_norm=40, norm_type=2),
    optimizer=dict(lr=0.001, momentum=0.9, type='SGD', weight_decay=0.0001),
    type='OptimWrapper')
param_scheduler = []
randomness = dict(deterministic=True, seed=20261017)
resume = False
train_cfg = dict(
    max_epochs=1, type='EpochBasedTrainLoop', val_begin=1, val_interval=1)
train_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file=
        'C:/Users/Dala/Documents/New project/mmaction_rebuild/configs/../data/splits/train.txt',
        data_prefix=dict(
            img=
            'C:/Users/Dala/Documents/New project/mmaction_rebuild/configs/../data/ucf101/rawframes'
        ),
        filename_tmpl='img_{:05}.jpg',
        modality='RGB',
        pipeline=[
            dict(
                clip_len=8, frame_interval=2, num_clips=1,
                type='SampleFrames'),
            dict(type='RawFrameDecode'),
            dict(scale=(
                -1,
                128,
            ), type='Resize'),
            dict(size=112, type='RandomCrop'),
            dict(input_format='NCTHW', type='FormatShape'),
            dict(type='PackActionInputs'),
        ],
        start_index=1,
        type='RawframeDataset'),
    num_workers=0,
    persistent_workers=False,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(clip_len=8, frame_interval=2, num_clips=1, type='SampleFrames'),
    dict(type='RawFrameDecode'),
    dict(scale=(
        -1,
        128,
    ), type='Resize'),
    dict(size=112, type='RandomCrop'),
    dict(input_format='NCTHW', type='FormatShape'),
    dict(type='PackActionInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file=
        'C:/Users/Dala/Documents/New project/mmaction_rebuild/configs/../data/splits/val.txt',
        data_prefix=dict(
            img=
            'C:/Users/Dala/Documents/New project/mmaction_rebuild/configs/../data/ucf101/rawframes'
        ),
        filename_tmpl='img_{:05}.jpg',
        modality='RGB',
        pipeline=[
            dict(
                clip_len=8,
                frame_interval=2,
                num_clips=1,
                test_mode=True,
                type='SampleFrames'),
            dict(type='RawFrameDecode'),
            dict(scale=(
                -1,
                128,
            ), type='Resize'),
            dict(crop_size=112, type='CenterCrop'),
            dict(input_format='NCTHW', type='FormatShape'),
            dict(type='PackActionInputs'),
        ],
        start_index=1,
        test_mode=True,
        type='RawframeDataset'),
    num_workers=0,
    persistent_workers=False,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    metric_list=(
        'top_k_accuracy',
        'mean_class_accuracy',
    ),
    metric_options=dict(top_k_accuracy=dict(topk=(
        1,
        5,
    ))),
    type='AccMetric')
val_pipeline = [
    dict(
        clip_len=8,
        frame_interval=2,
        num_clips=1,
        test_mode=True,
        type='SampleFrames'),
    dict(type='RawFrameDecode'),
    dict(scale=(
        -1,
        128,
    ), type='Resize'),
    dict(crop_size=112, type='CenterCrop'),
    dict(input_format='NCTHW', type='FormatShape'),
    dict(type='PackActionInputs'),
]
visualizer = dict(
    type='ActionVisualizer', vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = 'C:/Users/Dala/Documents/New project/mmaction_rebuild/configs/../work_dirs/swin_tiny_1epoch'
