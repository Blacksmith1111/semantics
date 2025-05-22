# !usr/bin/env python
# -*- coding:utf-8 _*-
"""
@Author: Huiqiang Xie
@Modified by: Assistant
@File: performance.py
@Time: 2021/4/1 11:48
"""
import os
import json
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from dataset import EurDataset, collate_data
from models.transceiver import DeepSC
from torch.utils.data import DataLoader
from utils import BleuScore, SNR_to_noise, greedy_decode, SeqtoText
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
from w3lib.html import remove_tags

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', default='europarl/train_data.pkl', type=str)
parser.add_argument('--vocab-file', default='europarl/vocab.json', type=str)
parser.add_argument('--checkpoint-path', default='checkpoints/deepsc-Rayleigh', type=str)
parser.add_argument('--channel', default='Rayleigh', type=str)
parser.add_argument('--MAX-LENGTH', default=30, type=int)
parser.add_argument('--MIN-LENGTH', default=4, type=int)
parser.add_argument('--d-model', default=128, type=int)
parser.add_argument('--dff', default=512, type=int)
parser.add_argument('--num-layers', default=4, type=int)
parser.add_argument('--num-heads', default=8, type=int)
parser.add_argument('--batch-size', default=64, type=int)
parser.add_argument('--epochs', default=1, type=int)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Similarity():
    def __init__(self, model_name='paraphrase-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def compute_similarity(self, real, predicted):
        real = [remove_tags(s) for s in real]
        predicted = [remove_tags(s) for s in predicted]
        embeddings1 = self.model.encode(real, convert_to_tensor=True)
        embeddings2 = self.model.encode(predicted, convert_to_tensor=True)
        cosine_scores = util.cos_sim(embeddings1, embeddings2)
        return cosine_scores.diagonal().cpu().numpy().tolist()


def plot_bleu_vs_snr(SNR, bleu_scores, output_dir='results'):
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    n_grams = len(bleu_scores)
    for i in range(n_grams):
        plt.plot(SNR, bleu_scores[i], label=f'{i+1}-gram', marker='o')
    plt.xlabel("SNR (dB)")
    plt.ylabel("BLEU Score")
    plt.title("BLEU Score vs SNR")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "bleu_vs_snr.png"))
    plt.close()


def plot_similarity_vs_snr(SNR, sim_scores, output_dir='results'):
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(SNR, sim_scores, label='Semantic Similarity', color='purple', marker='s')
    plt.xlabel("SNR (dB)")
    plt.ylabel("Cosine Similarity")
    plt.title("Semantic Similarity vs SNR")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "similarity_vs_snr.png"))
    plt.close()


def save_results(SNR, bleu_scores, sim_scores, output_dir='results'):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "scores.txt"), "w") as f:
        f.write("SNR\tBLEU-1\tBLEU-2\tBLEU-3\tBLEU-4\tSimilarity\n")
        for i in range(len(SNR)):
            line = [f"{SNR[i]:>3}"] + [f"{bleu_scores[j][i]:.4f}" for j in range(4)] + [f"{sim_scores[i]:.4f}"]
            f.write("\t".join(line) + "\n")


def performance(args, SNR, net):
    similarity = Similarity()
    bleu_scorers = [BleuScore(*([1 if i == j else 0 for j in range(4)])) for i in range(4)]

    test_eur = EurDataset('test')
    test_iterator = DataLoader(test_eur, batch_size=args.batch_size, num_workers=0,
                               pin_memory=True, collate_fn=collate_data)

    StoT = SeqtoText(token_to_idx, end_idx)
    bleu_scores_all = [[] for _ in range(4)]
    sim_scores_all = []

    net.eval()
    with torch.no_grad():
        for snr in tqdm(SNR, desc="Evaluating at SNRs"):
            noise_std = SNR_to_noise(snr)
            pred_sentences = []
            real_sentences = []

            for sents in test_iterator:
                sents = sents.to(device)
                target = sents

                out = greedy_decode(net, sents, noise_std, args.MAX_LENGTH, pad_idx,
                                    start_idx, args.channel)

                pred = out.cpu().numpy().tolist()
                real = target.cpu().numpy().tolist()

                pred_sentences += list(map(StoT.sequence_to_text, pred))
                real_sentences += list(map(StoT.sequence_to_text, real))

            # BLEU scores
            for i, scorer in enumerate(bleu_scorers):
                score = np.mean(scorer.compute_blue_score(pred_sentences, real_sentences))

                #score = scorer.compute_blue_score(pred_sentences, real_sentences)
                bleu_scores_all[i].append(score)

            # Similarity
            sim = similarity.compute_similarity(pred_sentences, real_sentences)
            sim_scores_all.append(np.mean(sim))

    return bleu_scores_all, sim_scores_all


if __name__ == '__main__':
    args = parser.parse_args()
    SNR = [0, 3, 6, 9, 12, 15, 18]

    args.vocab_file = '/home/yuriy/.vscode-server/semantics/HT_4/DeepSC/' + args.vocab_file
    vocab = json.load(open(args.vocab_file, 'rb'))
    token_to_idx = vocab['token_to_idx']
    idx_to_token = dict(zip(token_to_idx.values(), token_to_idx.keys()))
    num_vocab = len(token_to_idx)
    pad_idx = token_to_idx["<PAD>"]
    start_idx = token_to_idx["<START>"]
    end_idx = token_to_idx["<END>"]

    deepsc = DeepSC(args.num_layers, num_vocab, num_vocab,
                    num_vocab, num_vocab, args.d_model, args.num_heads,
                    args.dff, 0.1).to(device)

    model_paths = []
    for fn in os.listdir(args.checkpoint_path):
        if fn.endswith('.pth'):
            idx = int(os.path.splitext(fn)[0].split('_')[-1])
            model_paths.append((os.path.join(args.checkpoint_path, fn), idx))

    model_paths.sort(key=lambda x: x[1])
    model_path, _ = model_paths[-1]
    checkpoint = torch.load(model_path)
    deepsc.load_state_dict(checkpoint)
    print('Model loaded!')

    bleu_scores, sim_scores = performance(args, SNR, deepsc)
    plot_bleu_vs_snr(SNR, bleu_scores)
    plot_similarity_vs_snr(SNR, sim_scores)
    save_results(SNR, bleu_scores, sim_scores)
    print("Evaluation complete. Results saved in 'results/' directory.")
