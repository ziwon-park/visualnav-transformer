#!/usr/bin/env python3
"""
Visual Nav Transformer 이미지 테스트 스크립트
사용자의 이미지로 모델을 테스트할 수 있습니다.

사용법:
    python test_with_images.py --model vint --topomap-dir my_test_photos \
                                --obs-images img1.jpg img2.jpg img3.jpg img4.jpg img5.jpg \
                                --goal-idx 5

또는 단일 이미지 쌍 테스트:
    python test_with_images.py --model vint \
                                --current-image current.jpg \
                                --goal-image goal.jpg
"""

import os
import sys
import torch
import numpy as np
import argparse
import yaml
from PIL import Image
from PIL import Image as PILImage

import matplotlib.pyplot as plt

# 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), 'deployment/src'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'train'))

from deployment.src.utils import load_model, transform_images
from vint_train.training.train_utils import get_action


def load_vint_model(model_name='vint'):
    """ViNT 모델 로드"""
    MODEL_CONFIG_PATH = "deployment/config/models.yaml"

    with open(MODEL_CONFIG_PATH, "r") as f:
        model_paths = yaml.safe_load(f)

    model_config_path = model_paths[model_name]["config_path"]
    with open(model_config_path, "r") as f:
        model_params = yaml.safe_load(f)

    ckpt_path = model_paths[model_name]["ckpt_path"]

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"모델 가중치를 찾을 수 없습니다: {ckpt_path}\n"
            f"Google Drive에서 다운로드하세요: "
            f"https://drive.google.com/drive/folders/1a9yWR2iooXFAqjQHetz263--4_2FFggg"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"디바이스: {device}")
    print(f"모델 로드 중: {ckpt_path}")

    model = load_model(ckpt_path, model_params, device)
    model = model.to(device)
    model.eval()

    return model, model_params, device


def load_image_context(image_paths, context_size, image_size):
    """이미지 컨텍스트 로드 및 전처리"""
    if len(image_paths) < context_size:
        # 컨텍스트가 부족하면 마지막 이미지 복제
        image_paths = image_paths + [image_paths[-1]] * (context_size - len(image_paths))
    elif len(image_paths) > context_size:
        # 마지막 context_size개만 사용
        image_paths = image_paths[-context_size:]

    pil_images = [Image.open(path).convert('RGB') for path in image_paths]

    # 이미지 전처리
    transformed = transform_images(pil_images, image_size, center_crop=False)

    return transformed


def test_navigation(args):
    """네비게이션 테스트 실행"""
    print("=" * 60)
    print("Visual Nav Transformer 이미지 테스트")
    print("=" * 60)

    # 모델 로드
    model, model_params, device = load_vint_model(args.model)
    context_size = model_params['context_size']
    image_size = model_params['image_size']

    print(f"\n모델 정보:")
    print(f"  - 이름: {args.model}")
    print(f"  - Context size: {context_size}")
    print(f"  - Image size: {image_size}")

    # 이미지 로드
    if args.topomap_dir:
        # Topological map 사용
        topomap_dir = f"deployment/topomaps/images/{args.topomap_dir}"
        if not os.path.exists(topomap_dir):
            raise FileNotFoundError(f"Topomap 디렉토리를 찾을 수 없습니다: {topomap_dir}")

        topomap_files = sorted(
            [f for f in os.listdir(topomap_dir) if f.endswith(('.jpg', '.png'))],
            key=lambda x: int(os.path.splitext(x)[0])
        )
        topomap_paths = [os.path.join(topomap_dir, f) for f in topomap_files]

        print(f"\nTopological Map:")
        print(f"  - 경로: {topomap_dir}")
        print(f"  - 노드 수: {len(topomap_paths)}")

        # 관찰 이미지 (현재 위치)
        if args.obs_images:
            obs_paths = args.obs_images
        else:
            # 기본값: 처음 context_size개 이미지
            obs_paths = topomap_paths[:context_size]

        # 목표 이미지
        goal_idx = args.goal_idx if args.goal_idx != -1 else len(topomap_paths) - 1
        goal_path = topomap_paths[goal_idx]

        print(f"\n현재 위치 이미지: {[os.path.basename(p) for p in obs_paths]}")
        print(f"목표 이미지: {os.path.basename(goal_path)} (인덱스: {goal_idx})")

    else:
        # 단일 이미지 쌍 테스트
        if not args.current_image or not args.goal_image:
            raise ValueError(
                "--topomap-dir 또는 --current-image와 --goal-image를 지정해야 합니다"
            )

        obs_paths = [args.current_image] * context_size
        goal_path = args.goal_image

        print(f"\n현재 이미지: {args.current_image}")
        print(f"목표 이미지: {args.goal_image}")

    # 이미지 전처리
    print("\n이미지 전처리 중...")
    obs_images = load_image_context(obs_paths, context_size, image_size)
    goal_image = load_image_context([goal_path], 1, image_size)

    # 배치 차원 추가
    # obs_images = obs_images.unsqueeze(0).to(device)  # [1, context_size*3, H, W]
    # goal_image = goal_image.unsqueeze(0).to(device)  # [1, 3, H, W]
    # obs_images = obs_images.squeeze(1).unsqueeze(0).to(device)  # [1, 15, 64, 85]
    # goal_image = goal_image.squeeze(1).unsqueeze(0).to(device)  # [1, 3, 64, 85]
    obs_images = obs_images.view(1, 3 * context_size, image_size[0], image_size[1]).to(device)
    goal_image = goal_image.view(1, 3, image_size[0], image_size[1]).to(device)


    print(f"  - 관찰 이미지 텐서: {obs_images.shape}")
    print(f"  - 목표 이미지 텐서: {goal_image.shape}")

    # 추론
    print("\n모델 추론 중...")
    with torch.no_grad():
        # 거리 예측 (goal masking 모델의 경우)
        if model_params.get('goals_per_obs', 1) > 0:
            # 목표까지의 거리 예측
            # obs_cond_params = model('vision_encoder', obs_img=obs_images, goal_img=goal_image)
            obs_cond_params = model(obs_img=obs_images, goal_img=goal_image)

            # Waypoint 예측
            if 'nomad' in args.model:
                # NoMaD: Diffusion policy
                num_samples = args.num_samples
                print(f"  - Diffusion 샘플 수: {num_samples}")

                # TODO: NoMaD diffusion 샘플링 구현 필요
                # 현재는 기본 waypoint 예측만 수행
                waypoints = model('dist_pred_net', obsgoal_cond=obs_cond_params)
            else:
                # ViNT/GNM: 직접 waypoint 예측
                _, waypoints = model(obs_img=obs_images, goal_img=goal_image)
        else:
            # 기본 forward pass
            waypoints = model(obs_images, goal_image)

    # 결과 처리
    waypoints_np = waypoints.cpu().numpy()[0]  # [num_waypoints, 2]

    print("\n" + "=" * 60)
    print("예측 결과:")
    print("=" * 60)
    print(f"\nWaypoints (x, y 좌표):")
    for i, wp in enumerate(waypoints_np):
        print(f"  Waypoint {i}: x={wp[0]:.3f}, y={wp[1]:.3f}")

    # 시각화
    if args.visualize:
        visualize_results(obs_paths, goal_path, waypoints_np, args)

    return waypoints_np


def visualize_results(obs_paths, goal_path, waypoints, args):
    """결과 시각화"""
    print("\n결과 시각화 중...")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 현재 이미지
    current_img = Image.open(obs_paths[-1])
    axes[0].imshow(current_img)
    axes[0].set_title('현재 위치')
    axes[0].axis('off')

    # 목표 이미지
    goal_img = Image.open(goal_path)
    axes[1].imshow(goal_img)
    axes[1].set_title('목표 위치')
    axes[1].axis('off')

    # Waypoint 경로
    axes[2].plot(waypoints[:, 0], waypoints[:, 1], 'b-o', linewidth=2, markersize=8)
    axes[2].plot(0, 0, 'g*', markersize=15, label='시작 (로봇)')
    axes[2].plot(waypoints[-1, 0], waypoints[-1, 1], 'r*', markersize=15, label='목표')

    # Waypoint 번호 표시
    for i, wp in enumerate(waypoints):
        axes[2].annotate(f'{i}', (wp[0], wp[1]),
                        textcoords="offset points", xytext=(5,5),
                        fontsize=10, color='blue')

    axes[2].set_xlabel('X (m)')
    axes[2].set_ylabel('Y (m)')
    axes[2].set_title('예측된 경로')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    axes[2].axis('equal')

    plt.tight_layout()

    # 저장
    if args.output:
        plt.savefig(args.output, dpi=150, bbox_inches='tight')
        print(f"결과 저장: {args.output}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Visual Nav Transformer 이미지 테스트 스크립트'
    )

    # 모델 설정
    parser.add_argument(
        '--model',
        type=str,
        default='vint',
        choices=['vint', 'gnm', 'nomad'],
        help='사용할 모델 (기본값: vint)'
    )

    # Topomap 모드
    parser.add_argument(
        '--topomap-dir',
        type=str,
        help='Topological map 디렉토리 이름 (deployment/topomaps/images/ 내부)'
    )
    parser.add_argument(
        '--obs-images',
        nargs='+',
        help='관찰 이미지 경로 리스트 (현재 위치)'
    )
    parser.add_argument(
        '--goal-idx',
        type=int,
        default=-1,
        help='목표 노드 인덱스 (기본값: -1은 마지막 노드)'
    )

    # 단일 이미지 쌍 모드
    parser.add_argument(
        '--current-image',
        type=str,
        help='현재 위치 이미지 경로'
    )
    parser.add_argument(
        '--goal-image',
        type=str,
        help='목표 위치 이미지 경로'
    )

    # NoMaD 설정
    parser.add_argument(
        '--num-samples',
        type=int,
        default=8,
        help='NoMaD diffusion 샘플 수 (기본값: 8)'
    )

    # 시각화
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='결과 시각화 (matplotlib 필요)'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='시각화 결과 저장 경로 (예: result.png)'
    )

    args = parser.parse_args()

    # 테스트 실행
    try:
        waypoints = test_navigation(args)
        print("\n✅ 테스트 완료!")

    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
