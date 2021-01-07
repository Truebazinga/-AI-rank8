norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='CascadeEncoderDecoder',
    num_stages=2,
    pretrained='open-mmlab://msra/hrnetv2_w40',
    backbone=dict(
        type='HRNet',
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4, ),
                num_channels=(64, )),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                num_channels=(40, 80)),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(40, 80, 160)),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(40, 80, 160, 320)))),
    decode_head=[
        dict(
            type='FCNHead',
            in_channels=[40, 80, 160, 320],
            channels=sum([40, 80, 160, 320]),
            input_transform='resize_concat',
            in_index=(0, 1, 2, 3),
            kernel_size=1,
            num_convs=1,
            norm_cfg=norm_cfg,
            concat_input=False,
            dropout_ratio=-1,
            num_classes=7,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
        dict(
            type='OCRHead',
            in_channels=[40, 80, 160, 320],
            channels=512,
            ocr_channels=256,
            input_transform='resize_concat',
            in_index=(0, 1, 2, 3),
            norm_cfg=norm_cfg,
            dropout_ratio=-1,
            num_classes=7,
            align_corners=False,
            #loss_decode=dict(
            #    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0))
            loss_decode=dict(type='DiceLoss'))
    ])
train_cfg = dict(sampler=dict(type='OHEMPixelSampler'))
test_cfg = dict(mode='whole')
dataset_type = 'CityscapesDataset'
data_root = '/home/ma-user/work/combine/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 1024)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    #dict(type='Resize', img_scale=[(2048, 1024),(3072,1536),(2560,1280)], keep_ratio = True, multiscale_mode='value'),
    dict(type='Resize', img_scale=(2048, 1024), ratio_range=(0.5, 2.0)),
    #dict(type='RandomRotate',prob=0.3,degree=10),
    dict(type='RandomCrop', crop_size=(512, 1024), cat_max_ratio=0.75),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='PhotoMetricDistortion'),
    #dict(type='CLAHE'),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size=(512, 1024), pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 1024),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
    train=dict(
        type='CityscapesDataset',
        data_root='/home/ma-user/work/combine/',
        img_dir='leftImg8bit/train',
        ann_dir='gtFine/train',
        pipeline=train_pipeline),
    val=dict(
        type='CityscapesDataset',
        data_root='/home/ma-user/work/combine/',
        img_dir='leftImg8bit/val',
        ann_dir='gtFine/val',
        pipeline=test_pipeline),
    test=dict(
        type='CityscapesDataset',
        data_root='/home/ma-user/work/combine/',
        img_dir='leftImg8bit/val',
        ann_dir='gtFine/val',
        pipeline=test_pipeline))
log_config = dict(
    interval=100, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
#load_from = '/home/ma-user/work/mmseg-huawei/h40_ohem_rotate_combinedataset_multiscale_100ep/iter_70000.pth'
#load_from = '/home/ma-user/work/mmseg-huawei/h40_ohem_rotate_combinedataset_multiscale_100ep/iter_52500.pth'
resume_from = None
#resume_from='/home/ma-user/work/mmseg-huawei/h40_newdata_ohem_rotate_combinedataset_multiscale_100ep/iter_52000.pth'
workflow = [('train', 1)]
cudnn_benchmark = True
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()
lr_config = dict(policy='poly', power=0.9, min_lr=0.0001, by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=70000)
checkpoint_config = dict(by_epoch=False, interval=2000)
evaluation = dict(interval=2000, metric='mIoU')
work_dir = './h40_olddata_ohem_rotate_combinedataset_multiscale_100ep_0102'
gpu_ids = range(0, 1)


#optimizer = dict(type='SGD', lr=0.0002, momentum=0.9, weight_decay=0.0005)
#optimizer_config = dict()
#lr_config = dict(policy='poly', power=0.9, min_lr=0.000001, by_epoch=False)
#runner = dict(type='IterBasedRunner', max_iters=5000)
#checkpoint_config = dict(by_epoch=False, interval=500)
#evaluation = dict(interval=500, metric='mIoU')
#work_dir = './h40_ohem_rotate_combinedataset_multiscale_100ep_re_re'
#gpu_ids = range(0, 1)