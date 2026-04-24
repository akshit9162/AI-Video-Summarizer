import argparse
import os
import random
import sys

import numpy as np
import torch
import torch.optim as optim

if __package__ is None or __package__ == "":
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from src.dataset import load_videos
    from src.env import VideoSummarizationEnv
    from src.model import HorizontalPolicy, VerticalPolicy
    from src.video_utils import extract_features, extract_frames, importance
else:
    from src.dataset import load_videos
    from src.env import VideoSummarizationEnv
    from src.model import HorizontalPolicy, VerticalPolicy
    from src.video_utils import extract_features, extract_frames, importance


def discounted_returns(rewards, gamma):
    returns = []
    running = 0.0
    for r in reversed(rewards):
        running = float(r) + gamma * running
        returns.insert(0, running)
    returns = torch.tensor(returns, dtype=torch.float32)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    return returns


def save_checkpoint(path, h_policy, v_policy, config):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "horizontal_policy": h_policy.state_dict(),
            "vertical_policy": v_policy.state_dict(),
            "config": config,
        },
        path,
    )


def train(
    data_dir="data/videos",
    epochs=100,
    steps=40,
    k=25,
    gamma=0.99,
    lr=1e-4,
    entropy_coef=0.01,
    checkpoint_path="checkpoints/policies.pt",
    save_every=10,
):
    videos = load_videos(data_dir)
    if not videos:
        raise ValueError(f"No .mp4 videos found in {data_dir}")

    sample_frames = extract_frames(videos[0])
    feature_dim = extract_features(sample_frames).shape[1]

    h_policy = HorizontalPolicy(feature_dim, K=k)
    v_policy = VerticalPolicy(feature_dim)
    optimizer = optim.Adam(
        list(h_policy.parameters()) + list(v_policy.parameters()),
        lr=lr,
    )

    for epoch in range(1, epochs + 1):
        video_path = random.choice(videos)
        frames = extract_frames(video_path)
        feats = extract_features(frames)
        feats = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)
        imp = importance(feats)

        env = VideoSummarizationEnv(feats, imp, K=k)
        state = torch.tensor(env.reset(), dtype=torch.float32)

        rewards = []
        log_probs = []
        entropies = []

        for _ in range(steps):
            h_probs = h_policy(state)
            h_dist = torch.distributions.Categorical(h_probs)
            a_h = h_dist.sample()

            v_probs = v_policy(state, a_h.item())
            v_dist = torch.distributions.Categorical(v_probs)
            a_v = v_dist.sample()

            next_state, reward, _ = env.step(a_h.item(), a_v.item())
            state = torch.tensor(next_state, dtype=torch.float32)

            rewards.append(float(reward))
            log_probs.append(h_dist.log_prob(a_h) + v_dist.log_prob(a_v))
            entropies.append(h_dist.entropy() + v_dist.entropy())

        returns = discounted_returns(rewards, gamma)
        log_probs = torch.stack(log_probs)
        entropies = torch.stack(entropies)

        loss = -(log_probs * returns).mean() - entropy_coef * entropies.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % save_every == 0 or epoch == epochs:
            save_checkpoint(
                checkpoint_path,
                h_policy,
                v_policy,
                {
                    "feature_dim": feature_dim,
                    "k": k,
                    "epochs": epoch,
                    "steps": steps,
                    "gamma": gamma,
                },
            )

        print(
            f"Epoch {epoch}/{epochs} | "
            f"AvgReward: {np.mean(rewards):.4f} | Loss: {loss.item():.4f}"
        )

    print(f"Training complete. Checkpoint saved to {checkpoint_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/videos")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--k", type=int, default=25)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--checkpoint", default="checkpoints/policies.pt")
    parser.add_argument("--save-every", type=int, default=10)
    args = parser.parse_args()

    train(
        data_dir=args.data_dir,
        epochs=args.epochs,
        steps=args.steps,
        k=args.k,
        gamma=args.gamma,
        lr=args.lr,
        entropy_coef=args.entropy_coef,
        checkpoint_path=args.checkpoint,
        save_every=args.save_every,
    )