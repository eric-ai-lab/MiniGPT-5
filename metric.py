import os
import copy
from pathlib import Path
import json
import torch
import open_clip
from torchvision import transforms
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.fid import FrechetInceptionDistance
from PIL import Image
from tqdm import tqdm
from torchvision.transforms import ToPILImage, ToTensor
from sentence_transformers import SentenceTransformer, util
from rouge import Rouge 
from nltk.translate import bleu
from nltk.translate import meteor_score
from nltk import word_tokenize
from collections import OrderedDict
import numpy as np
from lightning.pytorch import seed_everything
import string
import argparse

from constants import *

class CLIPEvaluator(object):
    def __init__(self, device, name = 'ViT-L-14') -> None:
        self.device = device
        # self.model, self.preprocess = clip.load(clip_model, device=self.device)
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(name, pretrained='openai', device=device)
        self.tokenizer = open_clip.get_tokenizer(name)                                    # + skip convert PIL to tensor

    @torch.no_grad()
    def encode_text(self, tokens: list) -> torch.Tensor:
        return self.model.encode_text(tokens)

    @torch.no_grad()
    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        images = self.preprocess(images).unsqueeze(0).to(self.device)
        return self.model.encode_image(images)

    def get_text_features(self, text: str, norm: bool = True) -> torch.Tensor:

        tokens = self.tokenizer(text).to(self.device)

        text_features = self.encode_text(tokens).detach()

        if norm:
            text_features /= text_features.norm(dim=-1, keepdim=True)

        return text_features

    def get_image_features(self, img: torch.Tensor, norm: bool = True) -> torch.Tensor:
        image_features = self.encode_images(img)
        
        if norm:
            image_features /= image_features.clone().norm(dim=-1, keepdim=True)

        return image_features

    def img_to_img_similarity(self, src_images, generated_images):
        src_img_features = self.get_image_features(src_images)
        gen_img_features = self.get_image_features(generated_images)

        return (src_img_features @ gen_img_features.T).mean()

    def txt_to_img_similarity(self, text, generated_images):
        text_features    = self.get_text_features(text)
        gen_img_features = self.get_image_features(generated_images)

        return (text_features @ gen_img_features.T).mean()

def fid_preprocess_image(image):
    image = torch.tensor(image).unsqueeze(0)
    image = image.permute(0, 3, 1, 2) / 255.0
    return transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR)(image)

def cc3m_calculate_metrics(pred_folder):
    all_prediction_files = Path(pred_folder).glob("predictions-*.pt")
    predictions = []
    for file in all_prediction_files:
        predictions.extend(torch.load(file))

    clip_evaluator = CLIPEvaluator(device="cuda" if torch.cuda.is_available() else "cpu")
    inception = InceptionScore(normalize=True).cuda()
    basline_inception = InceptionScore(normalize=True).cuda()
    fid = FrechetInceptionDistance(normalize=True)
    to_pil = ToPILImage()

    fid_vis_processor = transforms.Compose(
                [
                    transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
                    transforms.ToTensor(),
                ]
            )
    
    # save true features
    fid_real_feaures_file = "fid_real_feaures.pt"
    if os.path.exists(fid_real_feaures_file):
        real_feaures = torch.load(fid_real_feaures_file)
        fid.real_features_sum = real_feaures[0]
        fid.real_features_cov_sum = real_feaures[1]
        fid.real_features_num_samples = real_feaures[2]
    else:
        image_paths = sorted([os.path.join(CC3M_FOLDER, x) for x in Path(CC3M_FOLDER).glob('val/*/*.jpg')])
        for path in tqdm(image_paths):
            image = Image.open(path).convert("RGB")
            image = fid_vis_processor(image).unsqueeze(0)
            fid.update(image, real=True)

        real_feaures = [fid.real_features_sum, fid.real_features_cov_sum, fid.real_features_num_samples]
        torch.save(real_feaures, "fid_real_feaures.pt")
    baseline_fid = copy.deepcopy(fid)

    clip_similarities = []
    clip_t_similarities = []
    basline_clip_similarities = []
    basline_clip_t_similarities = []

    for prediction in tqdm(predictions):
        _, _, gt_out, predicted_images_ft, predicted_images_nl, gt_image, _, _ = prediction
        if predicted_images_ft is not None:
            gt_image = to_pil(gt_image)
            gt_image_features = clip_evaluator.get_image_features(gt_image)
            gt_text_features = clip_evaluator.get_text_features(gt_out)
            predicted_images_ft_features = clip_evaluator.get_image_features(predicted_images_ft)
            predicted_images_nl_features = clip_evaluator.get_image_features(predicted_images_nl)

            clip_similarity = (gt_image_features @ predicted_images_ft_features.T).mean()
            clip_similarities.append(clip_similarity)

            clip_t_similarity = (gt_text_features @ predicted_images_ft_features.T).mean()
            clip_t_similarities.append(clip_t_similarity)

            basline_clip_similaritie = (gt_image_features @ predicted_images_nl_features.T).mean()
            basline_clip_similarities.append(basline_clip_similaritie)

            basline_clip_t_similarity = (gt_text_features @ predicted_images_nl_features.T).mean()
            basline_clip_t_similarities.append(basline_clip_t_similarity)

            totensor = ToTensor()
            inception.update(totensor(predicted_images_ft).unsqueeze(0).cuda())
            basline_inception.update(totensor(predicted_images_nl).unsqueeze(0).cuda())
            

            fid.update(fid_vis_processor(predicted_images_ft).unsqueeze(0), real=False)
            baseline_fid.update(fid_vis_processor(predicted_images_nl).unsqueeze(0), real=False)


    if len(clip_similarities) == 0:
        overall_clip_similarity = 0
        overall_basline_clip_similarity = 0
        overall_clip_t_similarity = 0
        overall_basline_clip_t_similarity = 0
    else:
        overall_clip_similarity = sum(clip_similarities) / len(clip_similarities)
        overall_basline_clip_similarity = sum(basline_clip_similarities) / len(basline_clip_similarities)
        overall_clip_t_similarity = sum(clip_t_similarities) / len(clip_t_similarities)
        overall_basline_clip_t_similarity = sum(basline_clip_t_similarities) / len(basline_clip_t_similarities)
    overall_fid_score = fid.compute().item()
    overall_basline_fid_score = baseline_fid.compute().item()
    overall_inception_score = inception.compute()[0].item()
    overall_basline_inception_score = basline_inception.compute()[0].item()

    if type(overall_clip_similarity) == torch.Tensor:
        overall_clip_similarity = overall_clip_similarity.item()
    if type(overall_inception_score) == torch.Tensor:
        overall_inception_score = overall_inception_score.item()
    if type(overall_basline_clip_similarity) == torch.Tensor:
        overall_basline_clip_similarity = overall_basline_clip_similarity.item()
    if type(overall_clip_t_similarity) == torch.Tensor:
        overall_clip_t_similarity = overall_clip_t_similarity.item()
    if type(overall_basline_clip_t_similarity) == torch.Tensor:
        overall_basline_clip_t_similarity = overall_basline_clip_t_similarity.item()

    print(f"Overall CLIP similarity: {overall_clip_similarity}")
    print(f"Overall CLIP text similarity: {overall_clip_t_similarity}")
    print(f"Overall Baseline CLIP similarity: {overall_basline_clip_similarity}")
    print(f"Overall Baseline CLIP text similarity: {overall_basline_clip_t_similarity}")
    print(f"Overall Inception Score: {overall_inception_score}")
    print(f"Overall Baseline Inception Score: {overall_basline_inception_score}")
    print(f"Overall FID score: {overall_fid_score}")
    print(f"Overall Baseline FID score: {overall_basline_fid_score}")


    results_dict = {
        'overall_clip_similarity': overall_clip_similarity,
        'overall_clip_t_similarity': overall_clip_t_similarity,
        'overall_basline_clip_similarity': overall_basline_clip_similarity,
        'overall_basline_clip_t_similarity': overall_basline_clip_t_similarity,
        'overall_inception_score': overall_inception_score,
        'overall_basline_inception_score': overall_basline_inception_score,
        'overall_fid_score': overall_fid_score,
        'overall_basline_fid_score': overall_basline_fid_score,
    }

    with open(os.path.join(pred_folder,'results_dict.json'), 'w') as f:
        json.dump(results_dict, f)

def vist_calculate_metrics(pred_folder, calculate_instace_level=True):
    all_prediction_files = Path(pred_folder).glob("predictions-*.pt")
    predictions = []
    for file in all_prediction_files:
        predictions.extend(torch.load(file))

    clip_evaluator = CLIPEvaluator(device="cuda" if torch.cuda.is_available() else "cpu")
    fid = FrechetInceptionDistance(normalize=True)
    baseline_fid = copy.deepcopy(fid)
    inception = InceptionScore(normalize=True).cuda()
    basline_inception = InceptionScore(normalize=True).cuda()

    to_pil = ToPILImage()
    totensor = ToTensor()

    fid_vis_processor = transforms.Compose(
                [
                    transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
                    transforms.ToTensor(),
                ]
            )
    with_basline = False

    clip_similarities = []
    clip_t_similarities = []
    basline_clip_similarities = []
    basline_clip_t_similarities = []
    fid_scores = []
    inception_scores = []
    task_dicts = {}
    for prediction in tqdm(predictions):
        _, _, gt_out, predicted_images_ft, predicted_images_nl, gt_image, _, task_name = prediction
        gt_image = to_pil(gt_image)
        task_name, step_id = task_name.split('_')
        if "-" in step_id:
            step_id = step_id.split('-')[0]
        step_id = int(step_id)
        if task_name not in task_dicts:
            task_dicts[task_name] = OrderedDict()
        if step_id not in task_dicts[task_name]:
            task_dicts[task_name][step_id] = {}
        task_dicts[task_name][step_id]['pred_out'] = predicted_images_ft
        task_dicts[task_name][step_id]['gt_out'] = gt_image
        if predicted_images_ft is not None:
            gt_image_features = clip_evaluator.get_image_features(gt_image)
            gt_text_features = clip_evaluator.get_text_features(gt_out)
            predicted_images_ft_features = clip_evaluator.get_image_features(predicted_images_ft)

            clip_similarity = (gt_image_features @ predicted_images_ft_features.T).mean()
            clip_similarities.append(clip_similarity)

            clip_t_similarity = (gt_text_features @ predicted_images_ft_features.T).mean()
            clip_t_similarities.append(clip_t_similarity)

            if calculate_instace_level:
                inception.update(totensor(predicted_images_ft).unsqueeze(0).cuda())
                

                fid.update(fid_vis_processor(predicted_images_ft).unsqueeze(0), real=False)
                fid.update(fid_vis_processor(gt_image).unsqueeze(0), real=True)
            
            if predicted_images_nl is not None:
                with_basline = True
                predicted_images_nl_features = clip_evaluator.get_image_features(predicted_images_nl)
                basline_clip_similaritie = (gt_image_features @ predicted_images_nl_features.T).mean()
                basline_clip_similarities.append(basline_clip_similaritie)

                basline_clip_t_similarity = (gt_text_features @ predicted_images_nl_features.T).mean()
                basline_clip_t_similarities.append(basline_clip_t_similarity)

                task_dicts[task_name][step_id]['pred_out_nl'] = predicted_images_nl

                if calculate_instace_level:
                    basline_inception.update(totensor(predicted_images_nl).unsqueeze(0).cuda())
                    baseline_fid.update(fid_vis_processor(predicted_images_nl).unsqueeze(0), real=False)

    if with_basline and calculate_instace_level:
        baseline_fid.real_features_sum = fid.real_features_sum
        baseline_fid.real_features_cov_sum = fid.real_features_cov_sum
        baseline_fid.real_features_num_samples = fid.real_features_num_samples
        overall_instance_basline_fid_score = baseline_fid.compute().item()
        overall_instance_basline_inception_score = basline_inception.compute()[0].item()
    else:
        overall_instance_basline_fid_score = 0
        overall_instance_basline_inception_score = 0

    if calculate_instace_level:
        overall_instance_fid_score = fid.compute().item()
        overall_instance_inception_score = inception.compute()[0].item()
    else:
        overall_instance_fid_score = 0
        overall_instance_inception_score = 0

    fid.reset()
    baseline_fid.reset()

    if len(clip_similarities) == 0:
        overall_clip_similarity = 0
        # overall_inception_score = 0
        overall_clip_t_similarity = 0
        overall_basline_clip_t_similarity = 0
    else:
        overall_clip_similarity = sum(clip_similarities) / len(clip_similarities)
        # overall_inception_score = sum(inception_scores) / len(inception_scores) 
        overall_clip_t_similarity = sum(clip_t_similarities) / len(clip_t_similarities)
    
    if len(basline_clip_similarities)==0:
        overall_basline_clip_similarity = 0
        overall_basline_clip_t_similarity = 0
    else:
        overall_basline_clip_similarity = sum(basline_clip_similarities) / len(basline_clip_similarities)
        overall_basline_clip_t_similarity = sum(basline_clip_t_similarities) / len(basline_clip_t_similarities)
    
    print(f"Overall CLIP similarity: {overall_clip_similarity}")
    print(f"Overall CLIP text similarity: {overall_clip_t_similarity}")
    print(f"Overall Baseline CLIP similarity: {overall_basline_clip_similarity}")
    print(f"Overall Baseline CLIP text similarity: {overall_basline_clip_t_similarity}")
    print(f"Overall Instance Inception Score: {overall_instance_inception_score}")
    print(f"Overall Instance Baseline Inception Score: {overall_instance_basline_inception_score}")
    print(f"Overall Instance FID score: {overall_instance_fid_score}")
    print(f"Overall Instance Baseline FID score: {overall_instance_basline_fid_score}")

    overall_fid_score = []
    overall_basline_fid_score = []
    for task in tqdm(task_dicts.values(), desc="Calculating task-level FID"):
        if len(task)==1:
            continue
        task = dict(sorted(task.items()))

        gt_images = [fid_vis_processor(t['gt_out']) for t in task.values()]
        gt_images = torch.stack(gt_images)
        fid.update(gt_images, real=True)

        predicted_images_ft = [fid_vis_processor(t['pred_out']) for t in task.values()]
        predicted_images_ft = torch.stack(predicted_images_ft)
        fid.update(predicted_images_ft, real=False)

        overall_fid_score.append(fid.compute().item())
        if 'pred_out_nl' in list(task.values())[0]:
            predicted_images_nl = [fid_vis_processor(t['pred_out_nl']) for t in task.values()]
            predicted_images_nl = torch.stack(predicted_images_nl)
            baseline_fid.update(predicted_images_nl, real=False)
            baseline_fid.real_features_sum = fid.real_features_sum
            baseline_fid.real_features_cov_sum = fid.real_features_cov_sum
            baseline_fid.real_features_num_samples = fid.real_features_num_samples
            overall_basline_fid_score.append(baseline_fid.compute().item())
        fid.reset()
        baseline_fid.reset()

    overall_fid_score = np.mean(overall_fid_score)
    overall_basline_fid_score = np.mean(overall_basline_fid_score)

    if type(overall_clip_similarity) == torch.Tensor:
        overall_clip_similarity = overall_clip_similarity.item()
    if type(overall_basline_clip_similarity) == torch.Tensor:
        overall_basline_clip_similarity = overall_basline_clip_similarity.item()
    if type(overall_clip_t_similarity) == torch.Tensor:
        overall_clip_t_similarity = overall_clip_t_similarity.item()
    if type(overall_basline_clip_t_similarity) == torch.Tensor:
        overall_basline_clip_t_similarity = overall_basline_clip_t_similarity.item()

    print(f"Overall FID score: {overall_fid_score}")
    print(f"Overall Baseline FID score: {overall_basline_fid_score}")


    results_dict = {
        'overall_clip_similarity': overall_clip_similarity,
        'overall_clip_t_similarity': overall_clip_t_similarity,
        'overall_basline_clip_similarity': overall_basline_clip_similarity,
        'overall_basline_clip_t_similarity': overall_basline_clip_t_similarity,
        'overall_fid_score': overall_fid_score,
        'overall_basline_fid_score': overall_basline_fid_score,
        'overall_instance_inception_score': overall_instance_inception_score,
        'overall_instance_basline_inception_score': overall_instance_basline_inception_score,
        'overall_instance_fid_score': overall_instance_fid_score,
        'overall_instance_basline_fid_score': overall_instance_basline_fid_score,
    }

    with open(os.path.join(pred_folder,'results_dict.json'), 'w') as f:
        json.dump(results_dict, f)

def mmdialog_calculate_metrics(pred_folder, calculate_instace_level=True):
    all_prediction_files = Path(pred_folder).glob("predictions-*.pt")
    predictions = []
    for file in all_prediction_files:
        predictions.extend(torch.load(file))

    clip_evaluator = CLIPEvaluator(device="cuda" if torch.cuda.is_available() else "cpu", name="ViT-B-32")
    
    inception = InceptionScore(normalize=True).cuda()
    rouge_model = Rouge()
    sentence_transformer_model = SentenceTransformer('all-MiniLM-L6-v2')

    to_pil = ToPILImage()
    totensor = ToTensor()

    bleu_weights = [(1.0,), (0.5, 0.5)]
    clip_similarities = []
    clip_t_similarities = []
    task_dicts = {}
    for prediction in tqdm(predictions):
        _, pred_out, gt_out, predicted_images_ft, predicted_images_nl, gt_image, _, task_name = prediction
        if '[IMG0]' in gt_out:
            need_image = True
        else:
            need_image = False
        
        # if '[IMG0]' in pred_out:
        if predicted_images_ft is not None:
            generate_image = True
        else:
            generate_image = False

        gt_out = gt_out.replace('###', '').split('[IMG0]')[0].strip()
        pred_out = pred_out.replace('###', '').split('[IMG0]')[0].strip()
        if all(c in string.punctuation for c in pred_out):
            pred_out = ''
        if all(c in string.punctuation for c in gt_out):
            gt_out = ''
        task_name, step_id = task_name.split('-')[0].split('_')
        step_id = int(step_id)
        if task_name not in task_dicts:
            task_dicts[task_name] = OrderedDict()
        if step_id not in task_dicts[task_name]:
            task_dicts[task_name][step_id] = {}
        if len(gt_out) and len(pred_out):
            bleu_score_1, bleu_score_2 = bleu([gt_out], pred_out, bleu_weights)
            rouge_scores = rouge_model.get_scores(pred_out, gt_out)
            rouge_l_score = rouge_scores[0]['rouge-l']['f']

            embeddings1 = sentence_transformer_model.encode(pred_out, convert_to_tensor=True, show_progress_bar=False)
            embeddings2 = sentence_transformer_model.encode(gt_out, convert_to_tensor=True, show_progress_bar=False)
            sbert_score = util.pytorch_cos_sim(embeddings1, embeddings2).item()

            task_dicts[task_name][step_id]['bleu_score_1'] = bleu_score_1
            task_dicts[task_name][step_id]['bleu_score_2'] = bleu_score_2
            task_dicts[task_name][step_id]['rouge_l_score'] = rouge_l_score
            task_dicts[task_name][step_id]['sbert_score'] = sbert_score

            gt_text_features = clip_evaluator.get_text_features(gt_out)
            pred_text_features = clip_evaluator.get_text_features(pred_out)
            clip_t_similarity = (gt_text_features @ pred_text_features.T).mean().item()

            task_dicts[task_name][step_id]['clip_t_similarity'] = clip_t_similarity
        
        elif len(gt_out):
            task_dicts[task_name][step_id]['bleu_score_1'] = 0
            task_dicts[task_name][step_id]['bleu_score_2'] = 0
            task_dicts[task_name][step_id]['rouge_l_score'] = 0
            task_dicts[task_name][step_id]['sbert_score'] = 0
            task_dicts[task_name][step_id]['clip_t_similarity'] = -1
        elif len(pred_out):
            task_dicts[task_name][step_id]['clip_t_similarity'] = -2
        
        if generate_image:
            inception.update(totensor(predicted_images_ft).unsqueeze(0).cuda())

        if need_image and generate_image:
            gt_image = to_pil(gt_image.float())
            gt_image_features = clip_evaluator.get_image_features(gt_image)
            predicted_images_ft_features = clip_evaluator.get_image_features(predicted_images_ft)
            clip_i_similarity = (gt_image_features @ predicted_images_ft_features.T).mean().item()

            task_dicts[task_name][step_id]['clip_i_similarity'] = clip_i_similarity
        elif need_image:
            task_dicts[task_name][step_id]['clip_i_similarity'] = -1
        elif generate_image:
            task_dicts[task_name][step_id]['clip_i_similarity'] = -2

    if calculate_instace_level:
        overall_instance_inception_score = inception.compute()[0].item()
    else:
        overall_instance_fid_score = 0
        overall_instance_inception_score = 0

    print(f"Overall Instance Inception Score: {overall_instance_inception_score}")

    overall_bleu_score_1 = []
    overall_bleu_score_2 = []
    overall_rouge_l_score = []
    overall_sbert_score = []
    overall_clip_f1 = []
    for task in tqdm(task_dicts.values(), desc="Calculating task-level FID"):
        task = dict(sorted(task.items()))

        task_bleu_score_1 = [t['bleu_score_1'] for t in task.values() if 'bleu_score_1' in t]
        task_bleu_score_2 = [t['bleu_score_2'] for t in task.values() if 'bleu_score_2' in t]
        task_rouge_l_score = [t['rouge_l_score'] for t in task.values() if 'rouge_l_score' in t]
        task_sbert_score = [t['sbert_score'] for t in task.values() if 'sbert_score' in t]

        task_clip_t_similarity = [t['clip_t_similarity'] for t in task.values() if 'clip_t_similarity' in t]
        task_clip_i_similarity = [t['clip_i_similarity'] for t in task.values() if 'clip_i_similarity' in t]

        task_clip_similarity = task_clip_t_similarity+task_clip_i_similarity
        valid_task_clip_similarity = [t for t in task_clip_similarity if t!=-1 and t!=-2]
        if len(valid_task_clip_similarity) == 0:
            clip_f1 = 0
        else:
            valid_clip_sum = sum(valid_task_clip_similarity)

            clip_precision = valid_clip_sum / len([t for t in task_clip_similarity if t!=-1])
            clip_recall = valid_clip_sum / len([t for t in task_clip_similarity if t!=-2])

            clip_f1 = 2 * clip_precision * clip_recall / (clip_precision + clip_recall)

        if len(task_bleu_score_1):
            overall_bleu_score_1.append(np.mean(task_bleu_score_1))
            overall_bleu_score_2.append(np.mean(task_bleu_score_2))
            overall_rouge_l_score.append(np.mean(task_rouge_l_score))
            overall_sbert_score.append(np.mean(task_sbert_score))
        overall_clip_f1.append(clip_f1)
    
    overall_bleu_score_1 = np.mean(overall_bleu_score_1)
    overall_bleu_score_2 = np.mean(overall_bleu_score_2)
    overall_rouge_l_score = np.mean(overall_rouge_l_score)
    overall_sbert_score = np.mean(overall_sbert_score)
    overall_clip_f1 = np.mean(overall_clip_f1)

    print(f"Overall BLEU-1 score: {overall_bleu_score_1}")
    print(f"Overall BLEU-2 score: {overall_bleu_score_2}")
    print(f"Overall ROUGE-L score: {overall_rouge_l_score}")
    print(f"Overall Sentence-BERT score: {overall_sbert_score}")
    print(f"Overall CLIP F1 score: {overall_clip_f1}")

    results_dict = {
        'overall_bleu_score_1': overall_bleu_score_1,
        'overall_bleu_score_2': overall_bleu_score_2,
        'overall_rouge_l_score': overall_rouge_l_score,
        'overall_sbert_score': overall_sbert_score,
        'overall_clip_f1': overall_clip_f1,
        'overall_instance_inception_score': overall_instance_inception_score,
    }

    with open(os.path.join(pred_folder,'results_dict.json'), 'w') as f:
        json.dump(results_dict, f)

if __name__=="__main__":
    seed_everything(42)
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_weight', type=str, help='an integer for the accumulator')
    args = parser.parse_args()
    test_weight = args.test_weight
    output_folder = os.path.join(OUTPUT_FOLDER, test_weight.split(".")[0]+OUTPUT_SUFFIX)
    if "CC3M" in DATAFOLDER:
        cc3m_calculate_metrics(output_folder)
    elif "MMDialog" in DATAFOLDER:
        mmdialog_calculate_metrics(output_folder)
    else:
        vist_calculate_metrics(output_folder)
    exit()