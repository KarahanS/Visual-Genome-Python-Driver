import numpy as np
from scipy import optimize


def bradley_terry_model(comparison_matrix, max_iter=100, tol=1e-8):
    """
    Convert pairwise comparison results to ratings using the Bradley-Terry model.

    Parameters:
    comparison_matrix (np.ndarray): An N x N matrix where entry (i,j) represents
                                  the number of times player i won against player j
    max_iter (int): Maximum number of iterations for the algorithm
    tol (float): Convergence tolerance

    Returns:
    np.ndarray: Array of ratings for each player
    """
    n_players = len(comparison_matrix)

    # Initialize ratings uniformly
    ratings = np.ones(n_players) / n_players

    # Get total games played by each player
    total_games = comparison_matrix + comparison_matrix.T

    # Get wins for each player
    wins = np.sum(comparison_matrix, axis=1)

    for iteration in range(max_iter):
        ratings_old = ratings.copy()

        # Calculate expected wins for each player
        expected_wins = np.zeros(n_players)
        for i in range(n_players):
            for j in range(n_players):
                if i != j and total_games[i, j] > 0:
                    expected_wins[i] += (
                        total_games[i, j] * ratings[i] / (ratings[i] + ratings[j])
                    )

        # Update ratings using maximum likelihood estimation
        for i in range(n_players):
            if wins[i] > 0:  # Only update if player has won any games
                ratings[i] = ratings[i] * wins[i] / expected_wins[i]

        # Normalize ratings to sum to 1
        ratings = ratings / np.sum(ratings)

        # Check convergence
        if np.max(np.abs(ratings - ratings_old)) < tol:
            break

    return ratings


def get_win_probability(rating_a, rating_b):
    """
    Calculate the probability of player A beating player B given their ratings.

    Parameters:
    rating_a (float): Rating of player A
    rating_b (float): Rating of player B

    Returns:
    float: Probability of player A winning
    """
    return rating_a / (rating_a + rating_b)


# Example usage
def example_usage():
    # Create sample comparison matrix
    # Entry (i,j) represents how many times player i beat player j
    comparison_matrix = np.array(
        [
            [0, 3, 4, 2],  # Player 0's wins against others
            [1, 0, 3, 2],  # Player 1's wins against others
            [0, 1, 0, 3],  # Player 2's wins against others
            [2, 2, 1, 0],  # Player 3's wins against others
        ]
    )

    # Calculate ratings
    ratings = bradley_terry_model(comparison_matrix)

    # Print results
    for i, rating in enumerate(ratings):
        print(f"Player {i} rating: {rating:.3f}")

    # Calculate win probability example
    prob = get_win_probability(ratings[0], ratings[1])
    print(f"\nProbability of Player 0 beating Player 1: {prob:.3f}")

    return ratings


if __name__ == "__main__":
    example_usage()
