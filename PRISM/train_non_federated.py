import copy
import gc
import random
from collections import defaultdict

import torch
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm

from dataloader import MovielensDatasetLoader, Movielens100kDatasetLoader, YelpDatasetLoader
from metrics import hit_ratio, ndcg, evaluate_hit_ratio, evaluate_ndcg, calculate_novelty, calculate_diversity
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


def federate(utils, selected_clients_indices):
    client_models = utils.get_user_models(utils.load_pytorch_client_model, selected_clients_indices)
    server_model = utils.get_previous_federated_model()
    if len(client_models) == 0:
        utils.save_federated_model(server_model)
        return
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


def update_server_model(utils, server_model, model):
    client_models = model
    if len(client_models) == 0:
        utils.save_federated_model(server_model)
        return
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
    return server_model


class FederatedNCF:
    def __init__(self, ui_matrix,  context_data, users_per_epoch, mode="ncf", aggregation_epochs=50,
                 local_epochs=10, batch_size=128, latent_dim=32, seed=0, use_context=False, context_dims=None,
                 differential_privacy_config=None, unseen_items=None):
        random.seed(seed)
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
        self.clients = self.generate_clients()
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

    def generate_clients(self):
        clients = []
        for user_idx in range(self.ui_matrix.shape[0]):
            client_data = self.ui_matrix[user_idx:user_idx + 1, :]  # Each client gets data of one user
            context_data = self.context_data[user_idx:user_idx + 1, :][0]
            clients.append(NCFTrainer(client_data, epochs=self.local_epochs, batch_size=self.batch_size,
                                      use_context=self.use_context, context_dims=self.context_dims,
                                      context_data=context_data))
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
            model = torch.jit.script(client.ncf.to(torch.device("cpu")))
            client_idx = selected_clients_indices[client_id]

            # Save the model before applying differential privacy
            torch.jit.save(model, "./models/local/pre_dp" + str(client_idx) + ".pt")

            if self.differential_privacy_config['state']:
                # Apply differential privacy to the model parameters
                model = self.apply_differential_privacy(model)

            torch.jit.save(model, "./models/local/dp" + str(client_idx) + ".pt")
            self.local_models[client_idx] = model
            bar.set_description(str(printing_single_round))
        bar.close()

    def apply_differential_privacy(self, model):
        """
        Applies differential privacy to the model parameters.
        """
        clip_norm = self.differential_privacy_config['clip_norm']
        delta = self.differential_privacy_config['delta']
        epsilon = self.differential_privacy_config['epsilon']

        noise_scale = np.sqrt(2 * np.log(1.25 / delta)) / epsilon

        for param in model.parameters():
            if param.grad is not None:
                # Compute the gradient norm
                grad_norm = param.grad.data.norm(2)

                # Clip the gradients
                if grad_norm > clip_norm:
                    param.grad.data.mul_(clip_norm / grad_norm)

                # Add Gaussian noise to the gradients
                noise = torch.normal(mean=0, std=noise_scale, size=param.grad.data.shape, device=param.grad.data.device,
                                     generator=torch.manual_seed(42))
                param.grad.data.add_(noise)

        return model

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
        random.seed(42)
        client_data = self.ui_matrix
        context_data = self.context_data[0]
        model = NCFTrainer(client_data, epochs=self.local_epochs, batch_size=self.batch_size,
                           use_context=self.use_context, context_dims=self.context_dims,
                           context_data=context_data)
        ncf_optimizer = torch.optim.Adam(model.ncf.parameters(), lr=5e-4)
        _, progress = model.train(ncf_optimizer, return_progress=True)
        model.train(self.ncf_optimizers[0])

        # self.server_model = update_server_model(self.utils, server_model, model)
        self.server_model = model

    def evaluate(self, test_data):
        test_ui_matrix, test_context_data = test_data
        hr_list, ndcg_list = [], []
        test_items_lists = defaultdict(list)
        recommendation_lists = defaultdict(list)
        test_item_ratings = []
        test_item_predictions = []

        for user_idx in range(test_ui_matrix.shape[0]):
            user_data = test_ui_matrix[user_idx:user_idx + 1, :]
            context_data = test_context_data[user_idx:user_idx + 1, :][0]
            client = self.server_model
            # client.ncf.load_server_weights(self.server_model)
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
                                        self.server_model.ncf.mlp_item_embeddings.weight)

        return avg_hr, avg_ndcg, avg_rmse, avg_mae, novelty, diversity


if __name__ == '__main__':
    differential_privacy_config = {
        'state': False,
        'epsilon': 10,
        'delta': 1e-5,
        'clip_norm': 1.0,
    }

    dataloader = Movielens100kDatasetLoader(loov=True)
    train_data, test_data = dataloader.train_data, dataloader.test_data
    fed_ncf = FederatedNCF(train_data[0], context_data=train_data[1],
                           users_per_epoch=int(dataloader.num_users*0.01), mode="ncf",
                           aggregation_epochs=20, local_epochs=8, batch_size=128,
                           use_context=True, context_dims=dataloader.context_dims,
                           differential_privacy_config=differential_privacy_config,
                           unseen_items=dataloader.unseen_items)
    fed_ncf.train()
    hr, ndcg, rmse, mae, novelty, diversity = fed_ncf.evaluate(dataloader.test_data)
    print(f'Loov Results: HR: {hr}, NDCG: {ndcg}, RMSE: {rmse}, MAE: {mae}, Novelty: {novelty}, Diversity: {diversity}')

    del fed_ncf, dataloader, train_data, test_data
    gc.collect()

    dataloader = Movielens100kDatasetLoader(loov=False)
    train_data, test_data = dataloader.train_data, dataloader.test_data
    fed_ncf = FederatedNCF(train_data[0], context_data=train_data[1],
                           users_per_epoch=int(dataloader.num_users * 0.01), mode="ncf",
                           aggregation_epochs=20, local_epochs=8, batch_size=128,
                           use_context=True, context_dims=dataloader.context_dims,
                           differential_privacy_config=differential_privacy_config,
                           unseen_items=dataloader.unseen_items)
    fed_ncf.train()
    hr, ndcg, rmse, mae, novelty, diversity = fed_ncf.evaluate(dataloader.test_data)
    print(f'Time Results: HR: {hr}, NDCG: {ndcg}, RMSE: {rmse}, MAE: {mae}, Novelty: {novelty}, Diversity: {diversity}')

    del fed_ncf, dataloader, train_data, test_data
    gc.collect()
