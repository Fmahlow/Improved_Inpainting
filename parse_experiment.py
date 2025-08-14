import argparse
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
from PIL import Image

try:
    import torch
    import clip
    from torchmetrics.image.fid import FrechetInceptionDistance
    import torchvision.transforms as T
except ImportError:
    torch = None
    clip = None
    FrechetInceptionDistance = None
    T = None

BACKGROUND_MAP: Dict[str, str] = {
    'The city of London': 'The city of London World',
    'The Parthenon in front of the Great Pyramid': 'The Parthenon in front of the Great Pyramid',
    'A single beam of light enters the room from the ceiling The beam of light is illuminating an easel On the easel there is a Rembrandt painting of a raccoon': 'A single beam of light enters the room from the ceiling. The beam of light is illuminating an easel. On the easel, there is a Rembrandt painting of a raccoon.',
    'A sunset': 'A sunset',
    'Photograph of a wall along a city street with a watercolor mural of foes in a jazz band': 'Photograph of a wall along a city street with a watercolor mural of foxes in a jazz band.'
}

CLOTHES_MAP: Dict[str, str] = {
    'A_scientist': 'A scientist',
    'A_photograph_of_a_knight_in_shining_armor_holding_a_basketball': 'A photograph of a knight in shining armor holding a basketball',
    'The_Mona_Lisa': 'The Mona Lisa',
    'Salvador_Dalí': 'Salvador Dalí',
    'A_person_with_arms_like_a_tree_branch': 'A person with arms like a tree branch'
}

REFERENCE_IMAGES: Dict[str, str] = {
    'Photograph of a wall along a city street with a watercolor mural of foxes in a jazz band.': 'reference_images/fox_mural.png',
    'The city of London World': 'reference_images/london.png',
    'The Parthenon in front of the Great Pyramid': 'reference_images/partenon_great_pyramid.png',
    'A sunset': 'reference_images/sunset.png',
    'A single beam of light enters the room from the ceiling. The beam of light is illuminating an easel. On the easel, there is a Rembrandt painting of a raccoon.': 'reference_images/racoon.png',
    'A scientist': 'reference_images/scientist.png',
    'A photograph of a knight in shining armor holding a basketball': 'reference_images/knight_basketball.png',
    'The Mona Lisa': 'reference_images/monalisa.jpg',
    'Salvador Dalí': 'reference_images/salvador_dali.jpeg',
    'A person with arms like a tree branch': 'reference_images/tree_arms.png'
}


def parse_threshold(th_str: str) -> float:
    return int(th_str[2:]) / 10.0


def compute_clip_score(image_path: Path, prompt: str, model, preprocess, device: str) -> float:
    if clip is None:
        raise ImportError("clip library is not installed")
    image = preprocess(Image.open(image_path).convert('RGB')).unsqueeze(0).to(device)
    text = clip.tokenize([prompt]).to(device)
    with torch.no_grad():
        img_feat = model.encode_image(image)
        txt_feat = model.encode_text(text)
        img_feat /= img_feat.norm(dim=-1, keepdim=True)
        txt_feat /= txt_feat.norm(dim=-1, keepdim=True)
        score = (img_feat @ txt_feat.T).squeeze().cpu().item()
    return float(score)

fid_transform = None


def setup_fid_transform() -> None:
    global fid_transform
    if T is None:
        raise ImportError("torchvision is not installed")
    fid_transform = T.Compose([
        T.Resize((299, 299)),
        T.ToTensor()
    ])


def compute_fid(image_path: Path, ref_path: Path, fid_metric) -> float:
    if fid_transform is None:
        setup_fid_transform()
    img = fid_transform(Image.open(image_path).convert('RGB')).unsqueeze(0)
    ref = fid_transform(Image.open(ref_path).convert('RGB')).unsqueeze(0)
    fid_metric.reset()
    fid_metric.update(ref, real=True)
    fid_metric.update(img, real=False)
    return float(fid_metric.compute().item())


def main(max_images: Optional[int] = None, output: str = 'experiment_metrics.csv') -> None:
    files = sorted(Path('experiment').glob('*.jpg'))
    if max_images:
        files = files[:max_images]

    device = 'cuda' if torch and torch.cuda.is_available() else 'cpu'
    if clip is None or FrechetInceptionDistance is None:
        raise ImportError('Required libraries for metrics not installed')
    model, preprocess = clip.load('ViT-B/32', device=device)
    fid_metric = FrechetInceptionDistance(feature=64)

    rows = []
    for file in files:
        parts = file.stem.split('__')
        background_raw = parts[2]
        th_raw = parts[3]
        clothing_raw = parts[4]
        background_prompt = BACKGROUND_MAP.get(background_raw, background_raw)
        clothing_prompt = CLOTHES_MAP.get(clothing_raw, clothing_raw.replace('_', ' '))
        text_prompt = f"{clothing_prompt} in {background_prompt}"

        clip_score = compute_clip_score(file, text_prompt, model, preprocess, device)

        back_ref = Path(REFERENCE_IMAGES[background_prompt])
        cloth_ref = Path(REFERENCE_IMAGES[clothing_prompt])
        fid_back = compute_fid(file, back_ref, fid_metric)
        fid_cloth = compute_fid(file, cloth_ref, fid_metric)
        fid_score = (fid_back + fid_cloth) / 2.0

        rows.append({
            'nome_arquivo': file.name,
            'imagem': parts[0],
            'threshold': parse_threshold(th_raw),
            'prompt_fundo': background_prompt,
            'prompt_roupa': clothing_prompt,
            'clip_score': clip_score,
            'fid': fid_score
        })

    df = pd.DataFrame(rows)
    df.to_csv(output, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute CLIP and FID metrics for experiment images')
    parser.add_argument('--max-images', type=int, default=None, help='limit number of images processed (for debugging)')
    parser.add_argument('--output', type=str, default='experiment_metrics.csv', help='output CSV file')
    args = parser.parse_args()
    main(max_images=args.max_images, output=args.output)
