import numpy as np

class VideoSummarizationEnv:
    def __init__(self, features, importance, K=25):
        self.features = features
        self.importance = importance
        self.N = len(features)
        self.K = K
        self.segment_size = max(1, self.N // self.K)

    def reset(self):
        self.selected_idx = np.sort(np.random.choice(self.N, self.K, replace=False))
        return self.features[self.selected_idx]

    def step(self, a_h, a_v):
        move = [-1, 1, -5, 5][a_v]
        idx = self.selected_idx[a_h]
        self.selected_idx[a_h] = np.clip(idx + move, 0, self.N - 1)
        self.selected_idx = np.sort(self.selected_idx)
        return self.features[self.selected_idx], self.reward(), False

    def reward(self):
        return np.tanh(
            0.4 * np.mean(self.importance[self.selected_idx]) +
            0.3 * self.diversity() +
            0.3 * self.coverage()
        )

    def diversity(self):
        feats = self.features[self.selected_idx]
        return np.mean([
            np.linalg.norm(f1 - f2)
            for f1 in feats for f2 in feats
        ])

    def coverage(self):
        return np.mean(np.diff(self.selected_idx)) / self.N