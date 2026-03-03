# ============================================================
# BEVFormer-Tiny for CARLA (Inference / Temporal Alignment Only)
# ============================================================

_base_ = [
    '../_base_/default_runtime.py'
]

custom_imports = dict(
    imports=[
        'projects.mmdet3d_plugin.datasets.carla_dataset',
        'projects.mmdet3d_plugin.datasets.pipelines'
    ],
    allow_failed_imports=False
)

plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'

# ------------------------------------------------------------
# Basic geometry
# ------------------------------------------------------------
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.2, 0.2, 8]

# ------------------------------------------------------------
# Image normalization
# ------------------------------------------------------------
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True
)

# ------------------------------------------------------------
# Input modality (CAMERA ONLY)
# ------------------------------------------------------------
input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True
)

# ------------------------------------------------------------
# BEVFormer-Tiny settings
# ------------------------------------------------------------
_dim_ = 256
_pos_dim_ = _dim_ // 2
_ffn_dim_ = _dim_ * 2
_num_levels_ = 1

bev_h_ = 50
bev_w_ = 50
queue_length = 3   # temporal frames

# ------------------------------------------------------------
# Model
# ------------------------------------------------------------
model = dict(
    type='BEVFormer',
    use_grid_mask=True,
    video_test_mode=True,
    pretrained=dict(img='torchvision://resnet50'),

    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3,),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch'
    ),

    img_neck=dict(
        type='FPN',
        in_channels=[2048],
        out_channels=_dim_,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=_num_levels_,
        relu_before_extra_convs=True
    ),

    pts_bbox_head=dict(
        type='BEVFormerHead',
        bev_h=bev_h_,
        bev_w=bev_w_,
        num_query=900,
        num_classes=10,
        in_channels=_dim_,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=False,

        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0
        ),

        transformer=dict(
            type='PerceptionTransformer',
            rotate_prev_bev=True,
            use_shift=True,
            use_can_bus=True,
            embed_dims=_dim_,

            encoder=dict(
                type='BEVFormerEncoder',
                num_layers=3,
                pc_range=point_cloud_range,
                num_points_in_pillar=4,
                return_intermediate=False,

                transformerlayers=dict(
                    type='BEVFormerLayer',
                    attn_cfgs=[
                        dict(
                            type='TemporalSelfAttention',
                            embed_dims=_dim_,
                            num_levels=1
                        ),
                        dict(
                            type='SpatialCrossAttention',
                            pc_range=point_cloud_range,
                            deformable_attention=dict(
                                type='MSDeformableAttention3D',
                                embed_dims=_dim_,
                                num_points=8,
                                num_levels=_num_levels_
                            ),
                            embed_dims=_dim_
                        )
                    ],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=(
                        'self_attn', 'norm',
                        'cross_attn', 'norm',
                        'ffn', 'norm'
                    )
                )
            ),

            decoder=dict(
                type='DetectionTransformerDecoder',
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=_dim_,
                            num_heads=8,
                            dropout=0.1
                        ),
                        dict(
                            type='CustomMSDeformableAttention',
                            embed_dims=_dim_,
                            num_levels=1
                        )
                    ],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=(
                        'self_attn', 'norm',
                        'cross_attn', 'norm',
                        'ffn', 'norm'
                    )
                )
            )
        ),

        bbox_coder=dict(
            type='NMSFreeCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
            max_num=300,
            voxel_size=voxel_size,
            num_classes=10
        ),

        positional_encoding=dict(
            type='LearnedPositionalEncoding',
            num_feats=_pos_dim_,
            row_num_embed=bev_h_,
            col_num_embed=bev_w_
        )
    ),

    # no training cfg needed for inference
    train_cfg=None,
    test_cfg=dict()
)

# ------------------------------------------------------------
# Dataset (CARLA)
# ------------------------------------------------------------
dataset_type = 'CarlaDataset'
data_root = '/media/fast/carla_runs/2026-02-06_16-39-19_6rgb_test'   # 改成你的路径

file_client_args = dict(backend='disk')

# ------------------------------------------------------------
# Pipeline (INFERENCE ONLY)
# ------------------------------------------------------------
test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(
        type='CustomDefaultFormatBundle3D',
        with_label=False
    ),
    dict(
        type='CustomCollect3D',
        keys=['img']
    )
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,

    # ⭐ 关键补充
    nonshuffler_sampler=dict(type='DistributedSampler'),

    test=dict(
        type=dataset_type,
        data_root=data_root,
        pipeline=test_pipeline,
        bev_size=(bev_h_, bev_w_),
        queue_length=queue_length,
        test_mode=True
    )
)

# ------------------------------------------------------------
# Runtime
# ------------------------------------------------------------
evaluation = dict(interval=1)
runner = dict(type='IterBasedRunner', max_iters=1)

log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook')]
)

checkpoint_config = dict(interval=1)
