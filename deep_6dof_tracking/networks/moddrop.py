import torch
import random


class ModDrop(torch.nn.Module):
    def __init__(self, channel_groups, dropout_proba=0.5, normalize_channels=True):
        """

        :param channel_groups: Structure of channel to drop : e.g. : [[0, 1, 2], [3]]
        dropout will either remove channel 0,1,2 or 3
        :param dropout_proba: Probability of applying the dropout
        """
        super(ModDrop, self).__init__()
        self.dropout_proba = dropout_proba
        self.channel_groups = channel_groups
        self.normalize_channels = normalize_channels

    def forward(self, x):
        n_channel = x.size(1)
        n_sample = x.size(0)

        if self.training:
            # todo remove this loop?
            for i in range(n_sample):
                if random.uniform(0, 1) < self.dropout_proba:
                    channel_group = random.choice(self.channel_groups)
                    for channel in channel_group:
                        x[i, channel, :, :] = 0

        # if a channel is dropped, adjust the gain
        if self.normalize_channels:
            channel_sums = torch.sum(x, dim=(2, 3))
            for i in range(x.size(0)):
                gain = 0
                for c in range(n_channel):
                    if channel_sums[i, c]:
                        gain += 1
                x[i, :, :, :] /= gain
        return x


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt

    layer = ModDrop([[0, 1, 2], [3]], 0.5)
    layer.train()

    for i in range(100):
        test_img = np.ones((32, 4, 50, 50), dtype=np.float32)
        test_img_torch = torch.from_numpy(test_img)

        out = layer(test_img_torch)

        plt.subplot(1, 5, 1)
        plt.imshow(out.numpy()[0, 0, :, :], vmin=0, vmax=1)
        plt.subplot(1, 5, 2)
        plt.imshow(out.numpy()[0, 1, :, :], vmin=0, vmax=1)
        plt.subplot(1, 5, 3)
        plt.imshow(out.numpy()[0, 2, :, :], vmin=0, vmax=1)
        plt.subplot(1, 5, 4)
        plt.imshow(out.numpy()[0, 3, :, :], vmin=0, vmax=1)
        plt.subplot(1, 5, 5)
        plt.imshow(np.sum(out.numpy()[0, :, :, :], axis=0), vmin=0, vmax=4)
        plt.show()
