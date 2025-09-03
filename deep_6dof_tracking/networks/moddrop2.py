import torch
import random
import torch.nn as nn


class ModDrop2(torch.nn.Module):
    def __init__(self, channel_groups, dropout_proba=0.5, conv_size=10, conv_stride=1, normalize_channels=True):
        """

        :param channel_groups: Structure of channel to drop : e.g. : [[0, 1, 2], [3]]
        dropout will either remove channel 0,1,2 or 3
        :param dropout_proba: Probability of applying the dropout
        """
        super(ModDrop2, self).__init__()
        self.dropout_proba = dropout_proba
        self.channel_groups = channel_groups
        self.normalize_channels = normalize_channels

        #Generate convs for each channel groups
        self.convs = nn.ModuleList()
        for group in channel_groups:
            layer = nn.Sequential(nn.Conv2d(len(group), conv_size, 3, conv_stride),
                                  nn.ELU(inplace=True),
                                  nn.BatchNorm2d(conv_size))
            self.convs.append(layer)

    def forward(self, x):
        n_sample = x.size(0)

        # Compute the features
        modality_features = []
        for i, channels in enumerate(self.channel_groups):
            a = self.convs[i](x[:, channels, :, :])
            modality_features.append(a)

        if self.training:
            # todo remove this loop? (in dropout they use a mask)
            for i in range(n_sample):
                if random.uniform(0, 1) < self.dropout_proba:
                    drop_channel_index = random.randint(0, len(modality_features)-1)
                    modality_features[drop_channel_index][i, :, :, :] = 0
        else:
            # We check if the input modality was gone, and put the computed features to zero
            # This will remove the offset caused by batchnorm...
            # Todo : at test time we don't want to compute the convs... and this is deadly ugly..
            # get minibach/channel sums
            chan_sums = torch.sum(x, dim=(2, 3))
            for i, channels in enumerate(self.channel_groups):
                # todo this loop could be optimized
                for j in range(n_sample):
                    # here we sum the channel group.. if they are all zeros, means that we zero out their features
                    if not torch.sum(chan_sums[j, channels]):
                        modality_features[i][:, :, :, :] = 0

        x = torch.cat(modality_features, dim=1)
        """
        import matplotlib.pyplot as plt
        plt.subplot(2, 3, 1)
        plt.imshow(x[0, 0, :, :].detach().cpu().numpy())
        plt.subplot(2, 3, 2)
        plt.imshow(x[0, -1, :, :].detach().cpu().numpy())
        plt.subplot(2, 3, 3)
        plt.imshow(torch.sum(x[0, :, :, :], dim=0).detach().cpu().numpy())
        plt.show()
        """

        return x


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt

    layer = ModDrop2([[0, 1, 2], [3]], 0.5)
    layer.train()

    for i in range(100):
        test_img = np.ones((32, 4, 50, 50), dtype=np.float32)
        test_img_torch = torch.from_numpy(test_img)

        out = layer(test_img_torch).detach()

        plt.subplot(1, 3, 1)
        plt.imshow(out.numpy()[0, 0, :, :], vmin=0, vmax=1)
        plt.subplot(1, 3, 2)
        plt.imshow(out.numpy()[0, -1, :, :], vmin=0, vmax=1)
        plt.subplot(1, 3, 3)
        plt.imshow(np.sum(out.numpy()[0, :, :, :], axis=0), vmin=0, vmax=4)
        plt.show()
