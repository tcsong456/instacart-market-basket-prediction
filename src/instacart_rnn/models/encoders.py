from torch import nn
from torch.nn import functional as F


class OneHotFeature(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.output_dim = num_classes

    def forward(self, x):
        return F.one_hot(
            x,
            num_classes=self.num_classes,
        ).float()


class ClampOneHotEncoding(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.output_dim = num_classes

    def forward(self, x):
        in_range = (x >= 0) & (x < self.num_classes)
        x_one_hot = F.one_hot(
            x.clamp(0, self.num_classes - 1),
            num_classes=self.num_classes,
        ).float()
        x_one_hot = x_one_hot * in_range.unsqueeze(-1)
        return x_one_hot


class DenseProductNameEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        output_dim: int,
    ):
        super().__init__()

        self.output_dim = output_dim
        self.vocab_size = vocab_size
        self.clamp_ohe = ClampOneHotEncoding(vocab_size)

        self.product_name_dense = nn.Linear(
            vocab_size,
            output_dim,
        )

    def forward(self, x):
        product_names = self.clamp_ohe(x)
        product_names = product_names.amax(dim=1)
        product_names = F.relu(self.product_name_dense(product_names))

        return product_names


class ReturnEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()

        self.output_dim = embedding_dim

        self.embedding = nn.Embedding(
            num_embeddings=num_embeddings, embedding_dim=embedding_dim
        )

    def forward(self, x):
        return self.embedding(x)


class UnsqueezeLastDim(nn.Module):
    output_dim = 1

    def forward(self, x):
        return x.float().unsqueeze(-1)


class NormalizeTensor(nn.Module):
    def __init__(self, divisor: int):
        super().__init__()

        self.divisor = divisor
        self.unsqueezer = UnsqueezeLastDim()
        self.output_dim = self.unsqueezer.output_dim

    def forward(self, x):
        return self.unsqueezer(x / self.divisor)
