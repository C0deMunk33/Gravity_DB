import pandas as pd
import torch

class GravityFieldAnalysis:
    def __init__(self, csv_path):
        self.data_tensor = self.load_csv(csv_path)

    @staticmethod
    def load_csv(csv_path):
        df = pd.read_csv(csv_path)
        return torch.tensor(df.values, dtype=torch.float32)

    @staticmethod
    def calculate_field_vectors_batch(data_tensor, dim_ranges, fixed_values, resolution=10):
        """
        Calculate the gravity field vectors over a specific range in specific dimensions using batch processing.
        Excludes data points with NaN values.

        :param data_tensor: Tensor of data points in the high-dimensional space.
        :param dim_ranges: Dictionary specifying the ranges for the dimensions to vary.
        :param fixed_values: Tensor of fixed values for the other dimensions.
        :param resolution: Number of points in the grid along each varying dimension.
        :return: Tensor of gravity vectors for each point in the grid.
        """
        # Exclude data points with NaN values
        valid_data_tensor = data_tensor[~torch.isnan(data_tensor).any(dim=1)]

        # Prepare the grid points for each dimension
        grid_points = [torch.linspace(start, end, resolution) for _, (start, end) in dim_ranges.items()]

        # Create a mesh grid and reshape to a batch of points
        mesh = torch.meshgrid(*grid_points)
        batch_points = torch.stack(mesh, dim=-1).reshape(-1, len(dim_ranges))

        # Broadcast the fixed values to match the batch size
        fixed_values_batch = fixed_values.repeat(batch_points.shape[0], 1)

        # Update the batch of points with varying dimensions
        for dim_index, dim in enumerate(dim_ranges.keys()):
            fixed_values_batch[:, dim] = batch_points[:, dim_index]

        # Compute gravity vectors for the entire batch
        diff = fixed_values_batch[:, None, :] - valid_data_tensor[None, :, :]
        distances = torch.norm(diff, dim=-1) + 1e-6
        gravity_influences = diff / distances[..., None].pow(2)
        total_gravity_vectors = gravity_influences.sum(dim=1)

        return total_gravity_vectors


    @staticmethod
    def sort_by_strength(field_vectors, grid_points):
        """
        Sort the calculated vector field by the strength of the vectors, keeping track of the initial coordinates.

        :param field_vectors: Tensor of gravity vectors for each point in the grid.
        :param grid_points: Tensor of coordinates for each point in the grid.
        :return: Sorted vectors along with their corresponding coordinates.
        """
        # Calculate the magnitude of each vector
        magnitudes = torch.norm(field_vectors, dim=1)

        # Pair each vector with its corresponding grid point
        paired = torch.cat([grid_points, field_vectors, magnitudes.unsqueeze(1)], dim=1)

        # Sort by magnitude (strength), descending order
        sorted_pairs = paired[paired[:, -1].sort(descending=True)[1]]

        return sorted_pairs

    @staticmethod
    def find_closest_n_data_points(data_tensor, target_vector, n_points):
        """
        Find the closest N data points in the dataset to the given target vector.

        :param data_tensor: Tensor of data points.
        :param target_vector: The target vector to compare to.
        :param n_points: Number of closest points to return.
        :return: The closest N data points.
        """
        differences = data_tensor - target_vector
        distances = torch.norm(differences, dim=1)
        closest_idxs = torch.topk(distances, k=n_points, largest=False).indices
        return data_tensor[closest_idxs]

    @staticmethod
    def complete_partial_point(partial_point, total_dimensions, placeholder_value=0.0):
        """
        Complete a partial point with a placeholder value for the unspecified dimensions.

        :param partial_point: Dictionary with specified dimensions and their values.
        :param total_dimensions: Total number of dimensions in the space.
        :param placeholder_value: Value to use for unspecified dimensions.
        :return: Completed point as a tensor.
        """
        full_point = torch.full((total_dimensions,), placeholder_value, dtype=torch.float32)
        for dim, value in partial_point.items():
            full_point[dim] = value
        return full_point

    def find_closest_points_to_strongest_vector(self, partial_point, n_points, dim_ranges, resolution=10):
        completed_point = self.complete_partial_point(partial_point, self.data_tensor.shape[1])
        field_vectors = self.calculate_field_vectors_batch(self.data_tensor, dim_ranges, completed_point, resolution)
        strengths = torch.norm(field_vectors, dim=1)
        strongest_vector = field_vectors[torch.argmax(strengths)]
        return self.find_closest_n_data_points(self.data_tensor, strongest_vector, n_points)

    def merge_new_data(self, new_csv_path):
        """
        Merge new data from a CSV file into the existing dataset, matching columns by name.

        :param new_csv_path: Path to the new CSV file.
        """
        new_df = pd.read_csv(new_csv_path)

        # Ensure the DataFrame representation is available
        if not hasattr(self, 'data_tensor_df'):
            self.data_tensor_df = pd.DataFrame(self.data_tensor.numpy())

        # Align columns by name and merge
        combined_df = pd.concat([self.data_tensor_df, new_df], axis=0, ignore_index=True, sort=False)

        # Handle missing values (e.g., fill with zeros or another placeholder)
        combined_df.fillna(0, inplace=True)

        # Update the data tensor
        self.data_tensor = torch.tensor(combined_df.values, dtype=torch.float32)
        # Update the DataFrame representation
        self.data_tensor_df = combined_df

# Example usage
csv_path = 'path_to_your_10_column_csv_file.csv'
gravity_analyzer = GravityFieldAnalysis(csv_path)

partial_point = {2: 0.5, 4: 0.7}  # Example partial point
n_points = 5  # Number of closest points to find
dim_ranges = {0: (0, 1), 1: (0, 1)}  # Example: vary first two dimensions

closest_points = gravity_analyzer.find_closest_points_to_strongest_vector(partial_point, n_points, dim_ranges)
print("Closest points:\n", closest_points)


# Existing dataset
csv_path = 'path_to_your_existing_csv_file.csv'
gravity_analyzer = GravityFieldAnalysis(csv_path)

# Merge new data
new_csv_path = 'path_to_your_new_csv_file.csv'
gravity_analyzer.merge_new_data(new_csv_path)
