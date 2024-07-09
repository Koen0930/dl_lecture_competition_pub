import torch
import torch.nn.functional as F


def retrieval(query_embeddings: torch.Tensor, database_embeddings: torch.Tensor, label_table: torch.Tensor) -> torch.Tensor:
    # query_embeddingsを正規化
    query_embeddings = F.normalize(query_embeddings, dim=-1)
    # database_embeddingsを正規化
    database_embeddings = F.normalize(database_embeddings, dim=-1)
    # query_embeddingsとdatabase_embeddingsの内積を計算
    scores = torch.matmul(query_embeddings, database_embeddings.T)
    class_scores = torch.matmul(scores, label_table)
    return class_scores