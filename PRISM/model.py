import torch
import torch.nn as nn


class NeuralCollaborativeFiltering(nn.Module):
    def __init__(self, user_num, item_num, predictive_factor=32):
        super(NeuralCollaborativeFiltering, self).__init__()
        self.mlp_user_embeddings = nn.Embedding(num_embeddings=user_num, embedding_dim=2 * predictive_factor)
        self.mlp_item_embeddings = nn.Embedding(num_embeddings=item_num, embedding_dim=2 * predictive_factor)
        self.gmf_user_embeddings = nn.Embedding(num_embeddings=user_num, embedding_dim=2 * predictive_factor)
        self.gmf_item_embeddings = nn.Embedding(num_embeddings=item_num, embedding_dim=2 * predictive_factor)
        self.mlp = nn.Sequential(
            nn.Linear(4 * predictive_factor, 2 * predictive_factor),
            nn.ReLU(),
            nn.Linear(2 * predictive_factor, predictive_factor),
            nn.ReLU(),
            nn.Linear(predictive_factor, predictive_factor // 2),
            nn.ReLU()
        )
        self.gmf_out = nn.Linear(2 * predictive_factor, 1)
        self.gmf_out.weight = nn.Parameter(torch.ones(1, 2 * predictive_factor))
        self.mlp_out = nn.Linear(predictive_factor // 2, 1)
        self.output_logits = nn.Linear(2 * predictive_factor, 1)
        self.model_blending = 0.5  # alpha parameter, equation 13 in the paper
        self.initialize_weights()
        self.join_output_weights()

    def initialize_weights(self):
        nn.init.normal_(self.mlp_user_embeddings.weight, std=0.01)
        nn.init.normal_(self.mlp_item_embeddings.weight, std=0.01)
        nn.init.normal_(self.gmf_user_embeddings.weight, std=0.01)
        nn.init.normal_(self.gmf_item_embeddings.weight, std=0.01)
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
        nn.init.kaiming_uniform_(self.gmf_out.weight, a=1)
        nn.init.kaiming_uniform_(self.mlp_out.weight, a=1)

    def forward(self, x):
        user_id, item_id = x[:, 0], x[:, 1]
        gmf_product = self.gmf_forward(user_id, item_id)
        mlp_output = self.mlp_forward(user_id, item_id)
        output = self.output_logits(torch.cat([gmf_product, mlp_output], dim=1)).view(-1)
        return 1 + 4 * torch.sigmoid(output)

    def gmf_forward(self, user_id, item_id):
        user_emb = self.gmf_user_embeddings(user_id)
        item_emb = self.gmf_item_embeddings(item_id)
        return torch.mul(user_emb, item_emb)

    def mlp_forward(self, user_id, item_id):
        user_emb = self.mlp_user_embeddings(user_id)
        item_emb = self.mlp_item_embeddings(item_id)
        return self.mlp(torch.cat([user_emb, item_emb], dim=1))

    def join_output_weights(self):
        w = nn.Parameter(
            torch.cat((self.model_blending * self.gmf_out.weight, (1 - self.model_blending) * self.mlp_out.weight),
                      dim=1))
        self.output_logits.weight = w

    @staticmethod
    def layer_setter(model, model_copy):
        for m, mc in zip(model.parameters(), model_copy.parameters()):
            mc.data[:] = m.data[:]

    def load_server_weights(self, server_model):
        self.layer_setter(server_model.mlp_item_embeddings, self.mlp_item_embeddings)
        self.layer_setter(server_model.gmf_item_embeddings, self.gmf_item_embeddings)
        self.layer_setter(server_model.mlp, self.mlp)
        self.layer_setter(server_model.gmf_out, self.gmf_out)
        self.layer_setter(server_model.mlp_out, self.mlp_out)
        self.layer_setter(server_model.output_logits, self.output_logits)


class ContextAwareNeuralCollaborativeFiltering(nn.Module):
    def __init__(self, user_num, item_num, predictive_factor=32, context_dims=2, context_data=None):
        super(ContextAwareNeuralCollaborativeFiltering, self).__init__()
        self.context_data = None  # Register context_data as a buffer
        self.mlp_user_embeddings = nn.Embedding(num_embeddings=user_num, embedding_dim=2 * predictive_factor)
        self.mlp_item_embeddings = nn.Embedding(num_embeddings=item_num, embedding_dim=2 * predictive_factor)
        self.gmf_user_embeddings = nn.Embedding(num_embeddings=user_num, embedding_dim=2 * predictive_factor)
        self.gmf_item_embeddings = nn.Embedding(num_embeddings=item_num, embedding_dim=2 * predictive_factor)
        self.mlp_context_embeddings = nn.Linear(context_dims, 2 * predictive_factor)  # Linear layer to reduce context dimensions
        # self.mlp = nn.Sequential(
        #     nn.Linear(6 * predictive_factor, 2 * predictive_factor),
        #     nn.ReLU(),
        #     nn.Linear(2 * predictive_factor, predictive_factor),
        #     nn.ReLU(),
        #     nn.Linear(predictive_factor, predictive_factor // 2),
        #     nn.ReLU()
        # )
        self.mlp = nn.Sequential(
            nn.Linear(6 * predictive_factor, 2 * predictive_factor),
            nn.ReLU(),
            nn.Linear(2 * predictive_factor, predictive_factor),
            nn.ReLU(),
            nn.Linear(predictive_factor, predictive_factor // 2),
            nn.ReLU()
        )
        self.gmf_out = nn.Linear(2 * predictive_factor, 1)
        self.gmf_out.weight = nn.Parameter(torch.ones(1, 2 * predictive_factor))
        self.mlp_out = nn.Linear(predictive_factor // 2, 1)
        self.output_logits = nn.Linear(2 * predictive_factor, 1)
        self.model_blending = 0  # alpha parameter, equation 13 in the paper
        self.initialize_weights()
        self.join_output_weights()

    def set_context_data(self, context_data):
        self.context_data = torch.tensor(context_data, dtype=torch.float32)
        self.register_buffer('context_data_buffer', self.context_data)

    def initialize_weights(self):
        nn.init.normal_(self.mlp_user_embeddings.weight, std=0.01)
        nn.init.normal_(self.mlp_item_embeddings.weight, std=0.01)
        nn.init.normal_(self.mlp_context_embeddings.weight, std=0.01)
        nn.init.normal_(self.gmf_user_embeddings.weight, std=0.01)
        nn.init.normal_(self.gmf_item_embeddings.weight, std=0.01)
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

        nn.init.kaiming_uniform_(self.gmf_out.weight, a=1)
        nn.init.kaiming_uniform_(self.output_logits.weight, a=1)

    def forward(self, x):
        user_id, item_id = x[:, 0], x[:, 1]
        # Extract context_vector for each item_id directly
        context_vector = self.context_data_buffer[item_id]
        gmf_product = self.gmf_forward(user_id, item_id)
        mlp_output = self.mlp_forward(user_id, item_id, context_vector)
        output = self.output_logits(torch.cat([gmf_product, mlp_output], dim=1)).view(-1)
        return 1 + 4 * torch.sigmoid(output)

    def gmf_forward(self, user_id, item_id):
        user_emb = self.gmf_user_embeddings(user_id)
        item_emb = self.gmf_item_embeddings(item_id)
        return torch.mul(user_emb, item_emb)

    def mlp_forward(self, user_id, item_id, context_vector):
        user_emb = self.mlp_user_embeddings(user_id)
        item_emb = self.mlp_item_embeddings(item_id)
        context_emb = self.mlp_context_embeddings(context_vector)
        updated_user_emb = torch.cat([user_emb, context_emb], dim=1)

        return self.mlp(torch.cat([updated_user_emb, item_emb], dim=1))

    def join_output_weights(self):
        w = nn.Parameter(
            torch.cat((self.model_blending * self.gmf_out.weight, (1 - self.model_blending) * self.mlp_out.weight),
                      dim=1))
        self.output_logits.weight = w

    @staticmethod
    def layer_setter(model, model_copy):
        for m, mc in zip(model.parameters(), model_copy.parameters()):
            mc.data[:] = m.data[:]

    def load_server_weights(self, server_model):
        self.layer_setter(server_model.mlp_item_embeddings, self.mlp_item_embeddings)
        self.layer_setter(server_model.gmf_item_embeddings, self.gmf_item_embeddings)
        self.layer_setter(server_model.mlp, self.mlp)
        self.layer_setter(server_model.gmf_out, self.gmf_out)
        self.layer_setter(server_model.mlp_out, self.mlp_out)
        self.layer_setter(server_model.output_logits, self.output_logits)


if __name__ == '__main__':
    ncf = ContextAwareNeuralCollaborativeFiltering(100, 100, 64)
    print(ncf)
