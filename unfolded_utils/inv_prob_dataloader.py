import torch


## Implement function to load data
## x, y = Ax + noise
## y^\prime = Ax^\prime + noise

## Custom dataloader for sliced Wasserstein training
def ip_loader(
    data,
    forward_map,  # Forward map operator: x -> Ax
    noise_y,  # Function: Batch_size realisations of observation noise
    batch_size,
    kwargs,
    total_iters
):
    # Get dataloaders (currently assume data is returned in form imgs, labels)
    loader = torch.utils.data.DataLoader(data, **kwargs)

    for i in range(total_iters):  # Returns data indefinitely
        # Get iterable datasets
        # Use independent shuffles of dataset for comparison data and true data
        # to avoid bias in the SW loss
        iter_train = iter(loader)
        iter_gen = iter(loader)

        # Loop over data
        for train_batch,_ in iter_train:
            # Sample random training points
            gen_batch = next(iter_gen)[0]
            # Sample observations
            obs_train_batch = forward_map(train_batch) + noise_y(train_batch)
            obs_gen_batch = forward_map(gen_batch) + noise_y(gen_batch)
            # Return data
            yield obs_gen_batch, torch.cat((obs_train_batch, train_batch), dim = -1)
