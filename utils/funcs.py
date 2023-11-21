import numpy as np
import torch
from utils.utils import print_log
import torch.nn.functional as F
import kornia as K
from utils.KCenterGreedy import KCenterGreedy


def embedding_concat(x, y, use_cuda):
    device = torch.device('cuda' if use_cuda else 'cpu')
    B, C1, H1, W1 = x.size()
    _, C2, H2, W2 = y.size()
    s = int(H1 / H2)
    x = F.unfold(x, kernel_size=s, dilation=1, stride=s)
    x = x.view(B, C1, -1, H2, W2)
    z = torch.zeros(B, C1 + C2, x.size(2), H2, W2).to(device)
    for i in range(x.size(2)):
        z[:, :, i, :, :] = torch.cat((x[:, :, i, :, :], y), 1)
    z = z.view(B, -1, H2 * W2)
    z = F.fold(z, kernel_size=s, output_size=(H1, W1), stride=s)
    return z

def mahalanobis_torch(u, v, cov):
    delta = u - v
    m = torch.dot(delta, torch.matmul(cov, delta))
    return torch.sqrt(m)


def get_rot_mat(theta):
    theta = torch.tensor(theta)
    return torch.tensor([[torch.cos(theta), -torch.sin(theta), 0],
                         [torch.sin(theta), torch.cos(theta), 0]])

def get_translation_mat(a, b):
    return torch.tensor([[1, 0, a],
                         [0, 1, b]])

def rot_img(x, theta):
    dtype =  torch.FloatTensor
    rot_mat = get_rot_mat(theta)[None, ...].type(dtype).repeat(x.shape[0],1,1)
    grid = F.affine_grid(rot_mat, x.size()).type(dtype)
    x = F.grid_sample(x, grid, padding_mode="reflection")
    return x

def translation_img(x, a, b):
    dtype =  torch.FloatTensor
    rot_mat = get_translation_mat(a, b)[None, ...].type(dtype).repeat(x.shape[0],1,1)
    grid = F.affine_grid(rot_mat, x.size()).type(dtype)
    x = F.grid_sample(x, grid, padding_mode="reflection")
    return x

def hflip_img(x):
    x = K.geometry.transform.hflip(x)
    return x


def rot90_img(x,k):
    # k is 0,1,2,3
    degreesarr = [0., 90., 180., 270., 360]
    degrees = torch.tensor(degreesarr[k])
    x = K.geometry.transform.rotate(x, angle = degrees, padding_mode='reflection')
    return x

def grey_img(x):
    x = K.color.rgb_to_grayscale(x)
    x = x.repeat(1, 3, 1,1)
    return x


def denormalization(x):
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)
    # x = (x.transpose(1, 2, 0) * 255.).astype(np.uint8)
    return x


def denorm(x):
    """Convert the range from [-1, 1] to [0, 1]."""
    out = (x + 1) / 2
    return out.clamp_(0, 1)

def maddern_transform(x,alpha):
    eps= 1e-7
    B,C,H,W = x.shape
    maddern = 0.5 + torch.log(x[:,1,:,:]+eps) - alpha * torch.log(x[:,2,:,:]+eps) - (1-alpha)*torch.log(x[:,0,:,:]+eps)
    x = maddern.view([B, 1, H, W])
    print("shape",x.shape)
    maddern_img_3_channels = torch.cat([x]*3, dim=1)
    return maddern_img_3_channels

def apply_augmentations(config, augment_support_img, support_img):
     # rotate img with small angle
    for angle in [-np.pi/4, -3 * np.pi/16, -np.pi/8, -np.pi/16, np.pi/16, np.pi/8, 3 * np.pi/16, np.pi/4]:
        rotate_img = rot_img(support_img, angle)
        augment_support_img = torch.cat([augment_support_img, rotate_img], dim=0)
    
    # translate img
    for a,b in [(0.2,0.2), (-0.2,0.2), (-0.2,-0.2), (0.2,-0.2), (0.1,0.1), (-0.1,0.1), (-0.1,-0.1), (0.1,-0.1)]:
        trans_img = translation_img(support_img, a, b)
        augment_support_img = torch.cat([augment_support_img, trans_img], dim=0)
    
    # hflip img
    flipped_img = hflip_img(support_img)
    augment_support_img = torch.cat([augment_support_img, flipped_img], dim=0)
    
    # rgb to grey img
    greyed_img = grey_img(support_img)
    augment_support_img = torch.cat([augment_support_img, greyed_img], dim=0)
    
    # rotate img in 90 degree
    for angle in [1,2,3]:
        rotate90_img = rot90_img(support_img, angle)
        augment_support_img = torch.cat([augment_support_img, rotate90_img], dim=0)
    augment_support_img = augment_support_img[torch.randperm(augment_support_img.size(0))]
    
    if config.dataset.include_maddern_transform is True:
        madder_transf_supp_img = maddern_transform(support_img, config.dataset.alpha)
        augment_support_img = torch.cat([augment_support_img, madder_transf_supp_img], dim=0)

    return augment_support_img

def nearest_neighbors(embedding, n_neighbors, memory_bank):
    """Nearest Neighbours using brute force method and euclidean norm.

    Args:
        embedding (Tensor): Features to compare the distance with the memory bank.
        n_neighbors (int): Number of neighbors to look at

    Returns:
        Tensor: Patch scores.
        Tensor: Locations of the nearest neighbor(s).
    """
    print(memory_bank.shape)
    distances = torch.cdist(embedding, memory_bank, p=2.0)  # euclidean norm
    if n_neighbors == 1:
        # when n_neighbors is 1, speed up computation by using min instead of topk
        patch_scores, locations = distances.min(1)
    else:
        patch_scores, locations = distances.topk(k=n_neighbors, largest=False, dim=1)
    return patch_scores, locations

def reshape_embedding(embedding):
        """Reshape Embedding.

        Reshapes Embedding to the following format:
        [Batch, Embedding, Patch, Patch] to [Batch*Patch*Patch, Embedding]

        Args:
            embedding (Tensor): Embedding tensor extracted from CNN features.

        Returns:
            Tensor: Reshaped embedding tensor.
        """
        embedding_size = embedding.size(1)
        embedding = embedding.permute(0, 2, 3, 1).reshape(-1, embedding_size)
        return embedding
    
def subsample_embedding(embedding, coreset_sampling_ratio):
        """Subsample embedding based on coreset sampling and store to memory.

        Args:
            embedding (np.ndarray): Embedding tensor from the CNN
            sampling_ratio (float): Coreset sampling ratio
        """

        # Coreset Subsampling
        sampler = KCenterGreedy(embedding=embedding, sampling_ratio=coreset_sampling_ratio)
        coreset = sampler.sample_coreset()
        memory_bank = coreset
        return memory_bank
    
def compute_anomaly_score(patch_scores, locations, embedding, memory_bank):
        """Compute Image-Level Anomaly Score.

        Args:
            patch_scores (Tensor): Patch-level anomaly scores
            locations: Memory bank locations of the nearest neighbor for each patch location
            embedding: The feature embeddings that generated the patch scores
        Returns:
            Tensor: Image-level anomaly scores
        """

        #Set num_neighbors by your self
        num_neighbors = 9
        # Don't need to compute weights if num_neighbors is 1
        if num_neighbors == 1:
            return patch_scores.amax(1)
        batch_size, num_patches = patch_scores.shape
        # 1. Find the patch with the largest distance to it's nearest neighbor in each image
        max_patches = torch.argmax(patch_scores, dim=1)  # indices of m^test,* in the paper
        # m^test,* in the paper
        max_patches_features = embedding.reshape(batch_size, num_patches, -1)[torch.arange(batch_size), max_patches]
        # 2. Find the distance of the patch to it's nearest neighbor, and the location of the nn in the membank
        score = patch_scores[torch.arange(batch_size), max_patches]  # s^* in the paper
        nn_index = locations[torch.arange(batch_size), max_patches]  # indices of m^* in the paper
        # 3. Find the support samples of the nearest neighbor in the membank
        nn_sample = memory_bank[nn_index, :]  # m^* in the paper
        # indices of N_b(m^*) in the paper
        print("inside compute_anomaly_score")
        _, support_samples = nearest_neighbors(nn_sample, n_neighbors=num_neighbors, memory_bank=memory_bank)
        # 4. Find the distance of the patch features to each of the support samples
        distances = torch.cdist(max_patches_features.unsqueeze(1), memory_bank[support_samples], p=2.0)
        print("distances",distances.shape)

        # 5. Apply softmax to find the weights
        weights = (1 - F.softmax(distances.squeeze(1), 1))[..., 0]
        # 6. Apply the weight factor to the score
        score = weights * score  # s in the paper
        print("score",score.shape)
        return score

class EarlyStop():
    """Used to early stop the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=20, verbose=True, delta=0, save_name="checkpoint.pt"):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            save_name (string): The filename with which the model and the optimizer is saved when improved.
                            Default: "checkpoint.pt"
        """
        self.patience = patience
        self.verbose = verbose
        self.save_name = save_name
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, optimizer, log):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer, log)
        elif score < self.best_score - self.delta:
            self.counter += 1
            print_log((f'EarlyStopping counter: {self.counter} out of {self.patience}'), log)
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer, log)
            self.counter = 0

        return self.early_stop

    def save_checkpoint(self, val_loss, model, optimizer, log):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print_log((f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...'),
                      log)
        state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
        torch.save(state, self.save_name)
        self.val_loss_min = val_loss
