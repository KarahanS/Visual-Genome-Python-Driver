import random
import csv
from collections import defaultdict
import numpy as np


def load_images(filename):
    """Load image IDs from the input file."""
    with open(filename, "r") as f:
        return [line.strip() for line in f if line.strip()]


def generate_comparisons(images, total_comparisons=100000, comparisons_per_session=250):
    """Generate pairwise comparisons with weighted random selection."""
    # Track how many times each image has been used
    usage_count = defaultdict(int)
    comparisons = []
    num_sessions = total_comparisons // comparisons_per_session

    for session in range(num_sessions):
        session_comparisons = []

        for _ in range(comparisons_per_session):
            # Calculate selection weights (inverse of usage count + 1 to avoid division by zero)
            weights = [1 / (usage_count[img] + 1) for img in images]
            weights = np.array(weights) / sum(weights)

            # Select two different images based on weights
            img1, img2 = np.random.choice(images, size=2, replace=False, p=weights)

            # Update usage counts
            usage_count[img1] += 1
            usage_count[img2] += 1

            session_comparisons.append([session + 1, img1, img2])

        comparisons.extend(session_comparisons)

    return comparisons


def save_comparisons(comparisons, output_file):
    """Save comparisons to a CSV file."""
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["session_id", "image1", "image2"])
        writer.writerows(comparisons)


def print_statistics(comparisons, images):
    """Print statistics about image usage."""
    usage_count = defaultdict(int)
    for _, img1, img2 in comparisons:
        usage_count[img1] += 1
        usage_count[img2] += 1

    print("\nImage Usage Statistics:")
    print(f"Total images: {len(images)}")
    print(
        f"Average appearances per image: {sum(usage_count.values()) / len(images):.2f}"
    )
    print(f"Min appearances: {min(usage_count.values())}")
    print(f"Max appearances: {max(usage_count.values())}")


def main():
    # Configuration
    input_file = "sample2.txt"
    output_file = "comparisons2.csv"
    total_comparisons = 100000
    comparisons_per_session = 250

    # Load images
    images = load_images(input_file)

    # Generate comparisons
    print("Generating comparisons...")
    comparisons = generate_comparisons(
        images, total_comparisons, comparisons_per_session
    )

    # Save results
    save_comparisons(comparisons, output_file)
    print(f"\nGenerated {len(comparisons)} comparisons")
    print(f"Saved to: {output_file}")

    # Print statistics
    print_statistics(comparisons, images)


if __name__ == "__main__":
    main()
