import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class BaseDatasetLoader:
    def __init__(self, filename, npy_file, context_file, test_size=0.2, random_state=42, loov=False):
        self.random_state = random_state
        self.loov = loov
        self.filename = filename
        self.npy_file = npy_file
        self.context_file = context_file
        self.rating_tuples = self.read_ratings()
        self.data = self.add_context(self.rating_tuples)
        self.num_users = int(max(self.data.T[0]))
        self.num_items = int(max(self.data.T[1]))
        self.context_dims = self.data.shape[1] - 4
        self.test_size = test_size
        self.train_data, self.test_data = self.split_data()
        self.unseen_items = self.generate_unseen_items()

    def read_ratings(self):
        raise NotImplementedError("Subclasses should implement this!")

    def add_context(self, data):
        # Extract hour and day of the week information from the timestamp
        timestamps = data[:, 3]
        datetimes = pd.to_datetime(timestamps, unit='s')

        hours = datetimes.hour
        days_of_week = datetimes.dayofweek
        days_of_month = datetimes.day
        months_of_year = datetimes.month
        years = datetimes.year

        # Define the time of day categories
        morning_hours = [6, 7, 8, 9, 10, 11]
        noon_hours = [12, 13]
        afternoon_hours = [14, 15, 16, 17]
        evening_hours = [18, 19, 20]
        night_hours = [21, 22, 23, 0, 1, 2, 3, 4, 5]

        # Create binary variables for each time of day category
        is_morning = np.isin(hours, morning_hours).astype(int)
        is_noon = np.isin(hours, noon_hours).astype(int)
        is_afternoon = np.isin(hours, afternoon_hours).astype(int)
        is_evening = np.isin(hours, evening_hours).astype(int)
        is_night = np.isin(hours, night_hours).astype(int)

        # Create binary variables for weekday vs weekend
        is_weekday = (days_of_week < 5).astype(int)
        is_weekend = (days_of_week >= 5).astype(int)

        # Define the seasons
        winter_months = [12, 1, 2]
        spring_months = [3, 4, 5]
        summer_months = [6, 7, 8]
        fall_months = [9, 10, 11]

        # Create binary variables for each season
        is_winter = np.isin(months_of_year, winter_months).astype(int)
        is_spring = np.isin(months_of_year, spring_months).astype(int)
        is_summer = np.isin(months_of_year, summer_months).astype(int)
        is_fall = np.isin(months_of_year, fall_months).astype(int)

        # Stack the new context features to the original data
        context_features = np.vstack((
            hours, days_of_week, days_of_month, months_of_year, years,
            is_morning, is_noon, is_afternoon, is_evening, is_night,
            is_weekday, is_weekend,
            is_winter, is_spring, is_summer, is_fall
        )).T
        # Keep all columns except the fourth one, and then add the context features
        # new_data = np.hstack((data[:, :3], data[:, 4:], context_features))
        new_data = np.hstack((data[:, :3], context_features))
        # new_data = np.hstack((data[:, :3], data[:, 4:]))  # Ignore time context for ablation study

        return new_data

    def generate_uic_matrix(self, data):
        # Initialize a 3D array for context features
        context_data = np.zeros((self.num_users, self.num_items, self.context_dims))

        # Initialize a 2D array for ratings
        rating_data = np.zeros((self.num_users, self.num_items))

        for row in data:
            user_idx = int(row[0]) - 1
            item_idx = int(row[1]) - 1
            rating = float(row[2])
            context_features = row[3:]  # Context features start from index 3

            # Assign context features to the corresponding position
            context_data[user_idx, item_idx] = context_features

            # Assign rating to the corresponding position
            rating_data[user_idx, item_idx] = rating

        return context_data, rating_data

    def split_data(self):
        train = []
        test = []

        for user_id in np.unique(self.data[:, 0]):
            # Filter data for the current user ID
            user_data = self.data[self.data[:, 0] == user_id]

            # Sort the user data by timestamp
            user_data_sorted = user_data[np.argsort(user_data[:, 3])]

            # Filter for ratings >= 4 & max(rating)
            max_rating = max(4, np.max(user_data_sorted[:, 2]))
            high_rating_data = user_data_sorted[user_data_sorted[:, 2] == max_rating]

            # Split the user data into training and testing sets
            if self.loov:
                # Select the latest item with the maximum rating, if no ratings >= 4, use the latest item
                test_item = high_rating_data[-1] if len(high_rating_data) > 0 else user_data_sorted[-1]

                test.append(test_item)
                train.extend(user_data_sorted[user_data_sorted[:, 1] != test_item[1]])

            else:
                test_size = int(0.2 * len(user_data_sorted))

                # Select the latest test_size items with the maximum rating
                test_items = high_rating_data[-test_size:] if len(high_rating_data) > 0 \
                    else user_data_sorted[-test_size:]

                test.extend(test_items)
                train.extend(user_data_sorted[~np.isin(user_data_sorted, test_items).all(axis=1)])

        train = np.array(train)
        test = np.array(test)

        # Remove the timestamp column before creating UIC matrix
        train = np.delete(train, 3, axis=1)
        test = np.delete(test, 3, axis=1)

        context_train, ratings_train = self.generate_uic_matrix(train)
        context_test, ratings_test = self.generate_uic_matrix(test)

        np.save(self.npy_file.replace('.npy', '_train.npy'), ratings_train)
        np.save(self.context_file.replace('.npy', '_train.npy'), context_train)
        np.save(self.npy_file.replace('.npy', '_test.npy'), ratings_test)
        np.save(self.context_file.replace('.npy', '_test.npy'), context_test)

        return (
            np.load(self.npy_file.replace('.npy', '_train.npy')),
            np.load(self.context_file.replace('.npy', '_train.npy'))), \
            (np.load(self.npy_file.replace('.npy', '_test.npy')),
             np.load(self.context_file.replace('.npy', '_test.npy')))

    def generate_unseen_items(self, items_num=99):
        train_ratings, _ = self.train_data
        test_ratings, _ = self.test_data

        unseen_items = {}
        all_items = set(range(self.num_items))

        for user_idx in range(self.num_users):
            seen_items = set(np.nonzero(train_ratings[user_idx])[0]).union(set(np.nonzero(test_ratings[user_idx])[0]))
            unseen_items[user_idx] = list(all_items - seen_items)

        unseen_item_lists = {}
        for user_idx in range(self.num_users):
            for item_idx in np.nonzero(test_ratings[user_idx])[0]:
                unseen_items_for_user = unseen_items[user_idx]
                if len(unseen_items_for_user) >= items_num:
                    np.random.seed(self.random_state)
                    unseen_sample = np.random.choice(unseen_items_for_user, items_num, replace=False)
                else:
                    np.random.seed(self.random_state)
                    unseen_sample = np.random.choice(unseen_items_for_user, len(unseen_items_for_user), replace=False)

                unseen_item_lists[(user_idx, item_idx)] = unseen_sample.tolist()

        return unseen_item_lists


class MovielensDatasetLoader(BaseDatasetLoader):
    def __init__(self, filename='datasets/ml-1m/ratings.dat', npy_file='datasets/ml-1m/ratings.npy',
                 context_file='datasets/ml-1m/context.npy', **kwargs):
        super().__init__(filename, npy_file, context_file, **kwargs)

    def read_ratings(self):
        ratings = open(self.filename, 'r').readlines()
        data = np.array([[int(i) for i in rating.split("::")] for rating in ratings])
        return data


class Movielens100kDatasetLoader(BaseDatasetLoader):
    def __init__(self, filename='datasets/ml-100k/u.data', npy_file='datasets/ml-100k/ratings.npy',
                 context_file='datasets/ml-100k/context.npy', **kwargs):
        super().__init__(filename, npy_file, context_file, **kwargs)

    def read_ratings(self):
        ratings = open(self.filename, 'r').readlines()
        data = np.array([[int(rating.split('\t')[0]),
                          int(rating.split('\t')[1]),
                          float(rating.split('\t')[2]),
                          int(rating.split('\t')[3])] for rating in ratings])
        return data


class YelpDatasetLoader(BaseDatasetLoader):
    def __init__(self, csv_file='datasets/yelp_fdata/yelp_fdata.csv', npy_file='datasets/yelp_fdata/ratings.npy',
                 context_file='datasets/yelp_fdata/context.npy', **kwargs):
        super().__init__(csv_file, npy_file, context_file, **kwargs)

    def read_ratings(self):
        data = pd.read_csv(self.filename)
        label_encoder_user = LabelEncoder()
        label_encoder_item = LabelEncoder()
        data['user'] = label_encoder_user.fit_transform(data['user'])
        data['item'] = label_encoder_item.fit_transform(data['item'])

        # Rename 'stars' to 'rating' and 'date' to 'timestamp'
        data.rename(columns={'user': 'user_id', 'item': 'item_id', 'stars': 'rating', 'date': 'timestamp'}, inplace=True)

        # Convert the 'timestamp' column to Unix time (seconds since epoch)
        data['timestamp'] = pd.to_datetime(data['timestamp']).astype('int64') // 10**9

        # Convert strings to numerical identifiers
        data['user_id'] = pd.factorize(data['user_id'])[0]
        data['item_id'] = pd.factorize(data['item_id'])[0]

        # Keep only the specified context columns (43 to 71)
        context_columns = data.columns[43:71]
        time_indices = [8, 9, 21, 22, 23]  # remove Time for ablation
        social_indices = [2, 3, 4, 5, 6, 7]
        occasion_indices = [11, 12, 13, 14]
        activity_indices = [0, 1, 10, 15, 16, 17, 18, 19, 20, 24, 25, 26, 27]
        context_columns = [col for i, col in enumerate(context_columns) if i in activity_indices]  # keep one group
        context_data = data[context_columns].values

        # Combine user, item, rating, timestamp, and context columns
        combined_data = np.hstack((data[['user_id', 'item_id', 'rating', 'timestamp']].values, context_data))
        # combined_data = data[['user_id', 'item_id', 'rating', 'timestamp']]

        return combined_data


if __name__ == '__main__':
    dataloader = YelpDatasetLoader()
