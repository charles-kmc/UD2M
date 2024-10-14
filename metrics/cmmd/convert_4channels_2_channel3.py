
def convert_to_three_channels(tensor):
    # Assuming tensor has shape (None, None, 4)
    return tensor[..., :3]  # Keep only the first 3 channels