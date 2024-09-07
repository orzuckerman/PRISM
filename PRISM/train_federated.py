import copy
import gc
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm

from dataloader import Movielens100kDatasetLoader, MovielensDatasetLoader, YelpDatasetLoader
from metrics import evaluate_hit_ratio, evaluate_ndcg, calculate_novelty, calculate_diversity
from server_model import ServerNeuralCollaborativeFiltering, ContextAwareServerNeuralCollaborativeFiltering
from train_single import NCFTrainer


class Utils:
    def __init__(self, num_clients, local_path="./models/local_items/", server_path="./models/central/"):
        self.epoch = 0
        self.num_clients = num_clients
        self.local_path = local_path
        self.server_path = server_path

    @staticmethod
    def load_pytorch_client_model(path):
        return torch.jit.load(path)

    def get_user_models(self, loader, selected_clients_indices):
        models = []
        for client_id in selected_clients_indices:
            models.append({'model': loader(self.local_path + "dp" + str(client_id) + ".pt")})
        return models

    def get_previous_federated_model(self):
        self.epoch += 1
        return torch.jit.load(self.server_path + "server" + str(self.epoch - 1) + ".pt")

    def save_federated_model(self, model):
        torch.jit.save(model, self.server_path + "server" + str(self.epoch) + ".pt")


def apply_differential_privacy(param, differential_privacy_config):
    """
    Applies differential privacy to a single parameter.
    """
    clip_norm = differential_privacy_config['clip_norm']
    delta = differential_privacy_config['delta']
    epsilon = differential_privacy_config['epsilon'] * differential_privacy_config['epsilon_scale']
    noise_scale = np.sqrt(2 * np.log(1.25 / delta)) / epsilon

    # Clip the parameter values
    param_norm = param.norm(2).item()
    if param_norm > clip_norm:
        param.mul_(clip_norm / param_norm)

    # Add Gaussian noise to the parameter
    noise = torch.normal(mean=0, std=noise_scale, size=param.shape, device=param.device)
    param.add_(noise)
    return param


def federate(utils, selected_clients_indices, differential_privacy_config):
    client_models = utils.get_user_models(utils.load_pytorch_client_model, selected_clients_indices)
    server_model = utils.get_previous_federated_model()

    if len(client_models) == 0:
        utils.save_federated_model(server_model)
        return

    for client_model in client_models:
        client_state_dict = client_model['model'].state_dict()
        if differential_privacy_config['state']:
            for k in client_state_dict.keys():
                client_state_dict[k] = apply_differential_privacy(client_state_dict[k], differential_privacy_config)
        client_model['model'].load_state_dict(client_state_dict)

    n = len(client_models)
    server_new_dict = copy.deepcopy(client_models[0]['model'].state_dict())

    for i in range(1, len(client_models)):
        client_dict = client_models[i]['model'].state_dict()
        for k in client_dict.keys():
            server_new_dict[k] += client_dict[k]

    for k in server_new_dict.keys():
        server_new_dict[k] = server_new_dict[k] / n

    server_model.load_state_dict(server_new_dict)
    utils.save_federated_model(server_model)


class FederatedNCF:
    def __init__(self, ui_matrix,  context_data, users_per_epoch, mode="ncf", aggregation_epochs=50,
                 local_epochs=10, batch_size=128, latent_dim=32, seed=0, use_context=False, context_dims=None,
                 differential_privacy_config=None, unseen_items=None):
        self.set_seed(seed)
        self.ui_matrix = ui_matrix
        self.context_data = context_data
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.users_per_epoch = users_per_epoch  # Number of clients to train in each epoch
        self.latent_dim = latent_dim
        self.mode = mode
        self.aggregation_epochs = aggregation_epochs
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.use_context = use_context
        self.context_dims = context_dims
        self.differential_privacy_config = differential_privacy_config
        self.clients = self.generate_clients(seed)
        self.ncf_optimizers = [torch.optim.Adam(client.ncf.parameters(), lr=5e-4) for client in self.clients]
        self.utils = Utils(self.ui_matrix.shape[0])
        self.unseen_items = unseen_items
        self.local_models = {}
        self.server_model = None
        self.interactions_data = self._prepare_interactions_data(ui_matrix)

    @staticmethod
    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)  # for multi-GPU setups
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    @staticmethod
    def _prepare_interactions_data(ui_matrix):
        interactions_data = defaultdict(list)
        for user_idx in range(ui_matrix.shape[0]):
            item_indices = ui_matrix[user_idx].nonzero()[0]
            interactions_data[user_idx] = item_indices.tolist()
        return interactions_data

    def generate_clients(self, seed):
        clients = []
        for user_idx in range(self.ui_matrix.shape[0]):
            client_data = self.ui_matrix[user_idx:user_idx + 1, :]  # Each client gets data of one user
            context_data = self.context_data[user_idx:user_idx + 1, :][0]
            clients.append(NCFTrainer(client_data, epochs=self.local_epochs, batch_size=self.batch_size,
                                      use_context=self.use_context, context_dims=self.context_dims,
                                      context_data=context_data, seed=seed))
        return clients

    def single_round(self, selected_clients, selected_optimizers, selected_clients_indices, epoch=0):
        single_round_results = {key: [] for key in ["num_users", "loss", "hit_ratio@10", "ndcg@10"]}
        bar = tqdm(enumerate(selected_clients), total=self.users_per_epoch)
        for client_id, client in bar:
            results = client.train(selected_optimizers[client_id])
            for k, i in results.items():
                single_round_results[k].append(i)
            printing_single_round = {"epoch": epoch}
            printing_single_round.update({k: round(sum(i) / len(i), 4) for k, i in single_round_results.items()})
            client_idx = selected_clients_indices[client_id]
            model = torch.jit.script(client.ncf.to(torch.device("cpu")))
            torch.jit.save(model, "./models/local/dp" + str(client_idx) + ".pt")
            self.local_models[client_idx] = model
            bar.set_description(str(printing_single_round))
        bar.close()

    def extract_item_models(self, selected_clients_indices):
        for client_id in selected_clients_indices:
            model = torch.jit.load("./models/local/dp" + str(client_id) + ".pt")
            if self.use_context:
                item_model = ContextAwareServerNeuralCollaborativeFiltering(item_num=self.ui_matrix.shape[1],
                                                                            predictive_factor=self.latent_dim)
            else:
                item_model = ServerNeuralCollaborativeFiltering(item_num=self.ui_matrix.shape[1],
                                                                predictive_factor=self.latent_dim)
            item_model.set_weights(model)
            item_model = torch.jit.script(item_model.to(torch.device("cpu")))
            torch.jit.save(item_model, "./models/local_items/dp" + str(client_id) + ".pt")

    def train(self):
        if self.use_context:
            server_model = ContextAwareServerNeuralCollaborativeFiltering(item_num=self.ui_matrix.shape[1],
                                                                          predictive_factor=self.latent_dim)
        else:
            server_model = ServerNeuralCollaborativeFiltering(item_num=self.ui_matrix.shape[1],
                                                              predictive_factor=self.latent_dim)
        server_model = torch.jit.script(server_model.to(torch.device("cpu")))
        torch.jit.save(server_model, "./models/central/server" + str(0) + ".pt")

        for epoch in range(self.aggregation_epochs):
            server_model = torch.jit.load("./models/central/server" + str(epoch) + ".pt", map_location=self.device)
            random.seed(epoch)
            selected_clients_indices = random.sample(range(self.ui_matrix.shape[0]), self.users_per_epoch)
            selected_clients = [self.clients[i] for i in selected_clients_indices]
            selected_optimizers = [self.ncf_optimizers[i] for i in selected_clients_indices]

            # Distribution
            for client in selected_clients:
                client.ncf.to(self.device)
                client.ncf.load_server_weights(server_model)

            # Training
            self.single_round(selected_clients, selected_optimizers, selected_clients_indices, epoch=epoch)

            # Extracting Item Weights for Aggregation
            self.extract_item_models(selected_clients_indices)

            # Aggregation
            federate(self.utils, selected_clients_indices, self.differential_privacy_config)

        self.server_model = server_model

    def evaluate(self, test_data):
        test_ui_matrix, test_context_data = test_data
        hr_list, ndcg_list = [], []
        test_items_lists = defaultdict(list)
        recommendation_lists = defaultdict(list)
        test_item_ratings = []
        test_item_predictions = []

        for user_idx in range(test_ui_matrix.shape[0]):
            user_data = test_ui_matrix[user_idx:user_idx + 1, :]
            client = self.clients[user_idx]
            client.ncf.load_server_weights(self.server_model)
            client.ncf.to(self.device)
            client.ncf.eval()

            for item_idx in user_data.nonzero()[1]:  # For each test item
                unseen_items = self.unseen_items[
                    (user_idx, item_idx)]  # Fetch unseen items for the specific test user-item tuple
                test_items = [item_idx] + unseen_items  # Test item + unseen items
                test_items_vector = torch.tensor(test_items, dtype=torch.long).to(self.device)
                user_id_vector = torch.zeros(len(test_items), dtype=torch.long).to(self.device)
                inputs = torch.stack((user_id_vector, test_items_vector), dim=1)
                with torch.no_grad():
                    y_pred = client.ncf(inputs)
                    y_true = torch.zeros_like(y_pred)
                    y_true[0] = 1  # The test item is the first one in the list

                    _, top_indices = torch.topk(y_pred, k=10, dim=0)
                    top_item_ids = test_items_vector[top_indices].cpu().numpy().tolist()

                hr_list.append(evaluate_hit_ratio(y_true.cpu().numpy(), y_pred.cpu().numpy()))
                ndcg_list.append(evaluate_ndcg(y_true.cpu().numpy(), y_pred.cpu().numpy()))

                # RMSE and MAE calculations
                test_item_ratings.append(float(test_data[0][user_idx][item_idx]))
                test_item_predictions.append(float(y_pred[0]))

                # Store the recommendations
                user_item_key = str(user_idx) + '_' + str(item_idx)
                test_items_lists[user_item_key] = item_idx
                recommendation_lists[user_item_key].extend(top_item_ids)

        avg_hr = np.mean(hr_list)
        avg_ndcg = np.mean(ndcg_list)
        avg_rmse = mean_squared_error(test_item_ratings, test_item_predictions, squared=False)
        avg_mae = mean_absolute_error(test_item_ratings, test_item_predictions)

        # Calculate novelty and diversity
        novelty = calculate_novelty(recommendation_lists, self.interactions_data)
        diversity = calculate_diversity(recommendation_lists, test_items_lists,
                                        self.server_model.mlp_item_embeddings.weight)

        recommendation_lists_df = pd.DataFrame(recommendation_lists).T
        recommendation_lists_df.to_csv('recommendation_lists_df.csv')

        return avg_hr, avg_ndcg, avg_rmse, avg_mae, novelty, diversity


if __name__ == '__main__':
    loov = True
    aggregation_epochs = 80
    dataloader = MovielensDatasetLoader(loov=loov)
    train_data, test_data = dataloader.train_data, dataloader.test_data
    num_users = dataloader.num_users
    context_dims = dataloader.context_dims
    unseen_items = dataloader.unseen_items
    del dataloader

    differential_privacy_config = {
        'state': True,
        'epsilon': 10,
        'epsilon_scale': 2000,
        'delta': 1e-5,
        'clip_norm': 1.0
    }

    fed_ncf = FederatedNCF(train_data[0], context_data=train_data[1],
                           users_per_epoch=int(num_users * 0.01), mode="ncf",
                           aggregation_epochs=aggregation_epochs, local_epochs=30, batch_size=128,
                           use_context=True, context_dims=context_dims,
                           differential_privacy_config=differential_privacy_config,
                           unseen_items=unseen_items)
    fed_ncf.train()
    hr, ndcg, rmse, mae, novelty, diversity = fed_ncf.evaluate(test_data)
    print(f'HR: {hr:.4f}, NDCG: {ndcg:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, '
          f'Novelty: {novelty:.4f}, Diversity: {diversity:.4f}')

    del fed_ncf
    gc.collect()
