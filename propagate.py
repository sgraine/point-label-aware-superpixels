##### Superpixel Generation #####
# This is main script that should be run, with all the specified parameters

# Load necessary modules
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import Adam, lr_scheduler
import numpy as np
from spixel_utils import *
from ssn import CNN
import os, argparse
from skimage.segmentation._slic import _enforce_label_connectivity_cython

import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image
import matplotlib.colors as mcolors

from torchvision import transforms
import torchmetrics

# This function takes the clusters and outputs the soft membership of each pixel to each of the clusters
def members_from_clusters(sigma_val_xy, sigma_val_cnn, XY_features, CNN_features, clusters):
    B, K, _ = clusters.shape
    sigma_array_xy = torch.full((B, K), sigma_val_xy, device=device)
    sigma_array_cnn = torch.full((B, K), sigma_val_cnn, device=device)
    
    clusters_xy = clusters[:,:,0:2]
    dist_sq_xy = torch.cdist(XY_features, clusters_xy)**2

    clusters_cnn = clusters[:,:,2:]
    dist_sq_cnn = torch.cdist(CNN_features, clusters_cnn)**2

    soft_memberships = F.softmax( (- dist_sq_xy / (2.0 * sigma_array_xy**2)) + (- dist_sq_cnn / (2.0 * sigma_array_cnn**2)) , dim = 2)                # shape = [B, N, K] 
    
    return soft_memberships

# Function to take the maximum class likelihood per pixel and enforces connectivity within regions
# This function also absorbs tiny segments into larger segments based on the 'min size' calculation
def enforce_connectivity(hard, H, W, K_max, connectivity = True):
    # INPUTS
    # 1. posteriors:    shape = [B, N, K]
    B = 1

    hard_assoc = torch.unsqueeze(hard, 0).detach().cpu().numpy()                                 # shape = [B, N]
    hard_assoc_hw = hard_assoc.reshape((B, H, W))    

    segment_size = (H * W) / (int(K_max) * 1.0)

    min_size = int(0.06 * segment_size)
    max_size = int(H*W*10)

    hard_assoc_hw = hard_assoc.reshape((B, H, W))
    
    for b in range(hard_assoc.shape[0]):
        if connectivity:
            spix_index_connect = _enforce_label_connectivity_cython(hard_assoc_hw[None, b, :, :], min_size, max_size, 0)[0]
        else:
            spix_index_connect = hard_assoc_hw[b,:,:]

    return spix_index_connect

# Write our new loss function to contain a term for the Distortion loss and a term for the Conflict loss
class CustomLoss(nn.Module):
    def __init__(self, clusters_init, N, XY_features, CNN_features, features_cat, labels, sigma_val_xy = 0.5, sigma_val_cnn = 0.5, alpha = 1, num_pixels_used = 1000):
        super(CustomLoss, self).__init__()
        self.alpha = alpha # Weighting for the distortion loss
        self.clusters=nn.Parameter(clusters_init, requires_grad=True)   # clusters (torch.FloatTensor: shape = [B, K, C])
        B, K, _ = self.clusters.shape

        self.N = N

        self.sigma_val_xy = sigma_val_xy
        self.sigma_val_cnn = sigma_val_cnn

        self.sigma_array_xy = torch.full((B, K), self.sigma_val_xy, device=device)
        self.sigma_array_cnn = torch.full((B, K), self.sigma_val_cnn, device=device)

        self.XY_features = XY_features
        self.CNN_features = CNN_features
        self.features_cat = features_cat

        self.labels = labels
        self.num_pixels_used = num_pixels_used

    def forward(self):
        # computes the distortion loss of the superpixels and also our novel conflict loss
        #
        # INPUTS:
        # 1) features:      (torch.FloatTensor: shape = [B, N, C]) defines for each image the set of pixel features

        # B is the batch dimension
        # N is the number of pixels
        # K is the number of superpixels

        # RETURNS:
        # 1) sum of distortion loss and conflict loss scaled by alpha (we use lambda in the paper but this means something else when coding)
        indexes = torch.randperm(self.N)[:self.num_pixels_used]

        ##################################### DISTORTION LOSS #################################################
        # Calculate the distance between pixels and superpixel centres by expanding our equation: (a-b)^2 = a^2-2ab+b^2 
        features_cat_select = self.features_cat[:,indexes,:]
        dist_sq_cat = torch.cdist(features_cat_select, self.clusters)**2

        # XY COMPONENT
        clusters_xy = self.clusters[:,:,0:2]

        XY_features_select = self.XY_features[:,indexes,:]
        dist_sq_xy = torch.cdist(XY_features_select, clusters_xy)**2

        # CNN COMPONENT
        clusters_cnn = self.clusters[:,:,2:]

        CNN_features_select = self.CNN_features[:,indexes,:]
        dist_sq_cnn = torch.cdist(CNN_features_select, clusters_cnn)**2

        B, K, _ = self.clusters.shape
        
        soft_memberships = F.softmax( (- dist_sq_xy / (2.0 * self.sigma_array_xy**2)) + (- dist_sq_cnn / (2.0 * self.sigma_array_cnn**2)) , dim = 2)                # shape = [B, N, K]  

        # The distances are weighted by the soft memberships
        dist_sq_weighted = soft_memberships * dist_sq_cat                                           # shape = [B, N, K] 

        distortion_loss = torch.mean(dist_sq_weighted)                                          # shape = [1]

        ###################################### CONFLICT LOSS ###################################################
        # print("labels", labels.shape)                                                         # shape = [B, 1, H, W]
        
        labels_reshape = self.labels.permute(0,2,3,1).float()                                   # shape = [B, H, W, 1]   

        # Find the indexes of the class labels larger than 0 (0 is means unknown class)
        label_locations = torch.gt(labels_reshape, 0).float()                                   # shape = [B, H, W, 1]
        label_locations_flat = torch.flatten(label_locations, start_dim=1, end_dim=2)           # shape = [B, N, 1]  

        XY_features_label = (self.XY_features * label_locations_flat)[0]                        # shape = [N, 2]
        non_zero_indexes = torch.abs(XY_features_label).sum(dim=1) > 0                          # shape = [N] 
        XY_features_label_filtered = XY_features_label[non_zero_indexes].unsqueeze(0)           # shape = [1, N_labelled, 2]
        dist_sq_xy = torch.cdist(XY_features_label_filtered, clusters_xy)**2                    # shape = [1, N_labelled, K]

        CNN_features_label = (self.CNN_features * label_locations_flat)[0]                      # shape = [N, 15]
        CNN_features_label_filtered = CNN_features_label[non_zero_indexes].unsqueeze(0)         # shape = [1, N_labelled, 15]
        dist_sq_cnn = torch.cdist(CNN_features_label_filtered, clusters_cnn)**2                 # shape = [1, N_labelled, K]

        soft_memberships = F.softmax( (- dist_sq_xy / (2.0 * self.sigma_array_xy**2)) + (- dist_sq_cnn / (2.0 * self.sigma_array_cnn**2)) , dim = 2)          # shape = [B, N_labelled, K]  
        soft_memberships_T = torch.transpose(soft_memberships, 1, 2)                            # shape = [1, K, N_labelled]

        labels_flatten = torch.flatten(labels_reshape, start_dim=1, end_dim=2)[0]               # shape = [N, 1]
        labels_filtered = labels_flatten[non_zero_indexes].unsqueeze(0)                         # shape = [1, N_labelled, 1] 

        # Use batched matrix multiplication to find the inner product between all of the pixels 
        innerproducts = torch.bmm(soft_memberships, soft_memberships_T)                         # shape = [1, N_labelled, N_labelled]

        # Create an array of 0's and 1's based on whether the class of both the pixels are equal or not
        # If they are the the same class, then we want a 0 because we don't want to add to the loss
        # If the two pixels are not the same class, then we want a 1 because we want to penalise this
        check_conflicts_binary = (~torch.eq(labels_filtered, torch.transpose(labels_filtered, 1, 2))).float()      # shape = [1, N_labelled, N_labelled]

        # Multiply these ones and zeros with the innerproduct array
        # Only innerproducts for pixels with conflicting labels will remain
        conflicting_innerproducts = torch.mul(innerproducts, check_conflicts_binary)           # shape = [1, N_labelled, N_labelled]

        # Find average of the remaining values for the innerproducts 
        # If we are using batches, then we add this value to our previous stored value for the points loss
        conflict_loss = torch.mean(conflicting_innerproducts)                                # shape = [1]

        return distortion_loss + self.alpha*conflict_loss, distortion_loss, self.alpha*conflict_loss

# We optimize our superpixel centre locations by minimizing our novel loss function
def optimize_spix(criterion, optimizer, scheduler, norm_val_x, norm_val_y, num_iterations=1000):
    
    best_clusters = criterion.clusters
    prev_loss = float("inf")

    for i in range(1,num_iterations):
        loss, distortion_loss, conflict_loss = criterion()

        # Every ten steps we clamp the X and Y locations of the superpixel centres to within the bounds of the image
        if i % 10 == 0:
            with torch.no_grad():
                clusters_x_temp = torch.unsqueeze(torch.clamp(criterion.clusters[0,:,0], 0, ((image_width-1)*norm_val_x)), dim=1)
                clusters_y_temp = torch.unsqueeze(torch.clamp(criterion.clusters[0,:,1], 0, ((image_height-1)*norm_val_y)), dim=1)
                clusters_temp = torch.unsqueeze(torch.cat((clusters_x_temp, clusters_y_temp, criterion.clusters[0,:,2:]), dim=1), dim=0)
            criterion.clusters.data.fill_(0)
            criterion.clusters.data += clusters_temp 

        if loss < prev_loss:
            best_clusters = criterion.clusters
            prev_loss = loss.item()

        loss.backward(retain_graph=True)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step(loss)

        for param_group in optimizer.param_groups:
            curr_lr = param_group['lr']

        if curr_lr < 0.001:
            break

    return best_clusters

# This function creates the RGB output of the augmented ground truth
def plot_propagated(save_path, propagated):
    ####### Function to plot the propagated labels in RGB ########
    # Assumes the propagation completed by the prop_to_unlabelled_spix_feat function

    if NUM_CLASSES == 35:
        # UCSD Mosaics
        colors = [[167, 18, 159], [180, 27, 92], [104, 139, 233], [49, 198, 135], [98, 207, 26], [118, 208, 133], [158, 118, 90], [12, 72, 166], [69, 79, 238], [81, 195, 49],[221, 236, 52], [160, 200, 222],[255, 63, 216], [16, 94, 7], [226, 47, 64], [183, 108, 5], 
            [55, 252, 193], [147, 154, 196], [233, 78, 165], [108, 25, 95], [184, 221, 46], [54, 205, 145], [14, 101, 210], [199, 232, 230], [66, 10, 103], [161, 228, 59], [108, 2, 104], [13, 49, 127], [186, 99, 38], [97, 140, 246], [44, 114, 202], [36, 31, 118], [146, 77, 143],
            [188, 100, 14],[131, 69, 63]]

        bgr=np.array(colors)/255.
        rgb = bgr[:,::-1]

    elif NUM_CLASSES == 12:
        # CSIRO Segmentation
        colors = [[0, 0, 0], [255, 0, 0], [255, 51, 255], [0, 255, 0], [255, 255, 51], [119, 119, 119], [0, 204, 204], [204, 255, 119], [255, 255, 255], [204, 204, 153], [255, 119, 0], [0, 0, 255]]

        rgb=np.array(colors)/255.
    else:
        print("We don't have a stored colour map for that quantity of classes, please specify - see the plot propagated function.")

    mymap = mcolors.LinearSegmentedColormap.from_list('my_colormap', rgb)
    mymap.set_bad(alpha=0) # set how the colormap handles 'bad' values
    plt.register_cmap(name='my_colormap', cmap=mymap)
    plt.set_cmap('my_colormap')
    norm = mcolors.Normalize(vmin=0, vmax=NUM_CLASSES-1)
    m = cm.ScalarMappable(norm=norm, cmap=mymap)
  
    color = m.to_rgba(propagated)

    fig = plt.figure(figsize=(20,20), facecolor='w', frameon=False)
    plt.axis('off')
    plt.imshow(color, alpha=1.0)
    plt.savefig(save_path+".jpg", bbox_inches='tight')
    plt.close()                      

# This function propagates the class of the most similar superpixel to superpixels which do not have a point label inside
# We want all pixels in our augmented ground truth to have an associated label
def prop_to_unlabelled_spix_feat(sparse_labels, connected, features_cnn, H, W):
    ##### Function to propagate the label of our labelled superpixels to unlabelled superpixels in the image #####

    features_cnn = features_cnn.detach().cpu().numpy()      # shape = [B, N, C]
    features_cnn = features_cnn[0]                          # shape = [N, C]
    features_cnn_reshape = np.reshape(features_cnn, (H,W, np.shape(features_cnn)[1]))       # shape = [H, W, C]

    spix_features = []

    # Find the average feature vector for each of our connected clusters (we no longer have the same clusters as from the optimiser)
    # Iterate through each superpixel and average the features in that area
    for spix in np.unique(connected):
        r, c = np.where(connected == spix)          
        features_curr_spix = features_cnn_reshape[(r,c)]    # shape = [X, C]  where X is the number of pixels in our 'spix' superpixel
        average_features = np.mean(features_curr_spix, axis=0, keepdims=True)      # shape = [1, C]
        average_features = np.squeeze(average_features)     # shape = [C,]
        temp = [spix]                                       # shape = [1,]  -  this is the index of the current superpixel
        temp.extend(average_features)                       # shape = [C+1]  -  we have the superpixel index as the first value and then concatenate the C features afterwards
        spix_features.append( temp )

    # Our array containing all superpixels and their average feature vectors
    spix_features = np.array(spix_features)                 # shape = [K_new, C+1]  - for each connected superpixel (could be different from the specified K), we have the index and features

    mask_np = np.array(sparse_labels)
    mask_np = np.squeeze(mask_np)                           # shape = [H, W]

    labels = []
    image_size = np.shape(mask_np)
    # Iterate through each pixel in the mask
    for x in range(image_size[0]):
        for y in range(image_size[1]):
            if mask_np[x,y]>0:
                spixel_num = connected[int(x), int(y)]
                labels.append( [mask_np[x,y]-1, spixel_num, x, y] ) # This is the class !
    
    # Array containing the labelled pixels - for each we have the label number, the index of the superpixel it falls inside and the x,y coordinate of the random point
    labels_array = np.array(labels)                         # shape = [num_points, 4]

    spix_labels = []
    # Iterate through the superpixels in our image
    for spix_i in range(len(np.unique(connected))):
        # If that superpixel is already labelled, then let's add that to our list of labelled superpixels
        spix = np.unique(connected)[spix_i]
        if spix in labels_array[:,1]:
            label_indices = np.where(labels_array[:,1] == spix)
            labels = labels_array[label_indices]
            most_common = np.argmax(np.bincount(labels[:,0]))
            temp = [spix, most_common]
            temp.extend(spix_features[spix_i,1:])
            spix_labels.append( temp )

    # Create a list of our LABELLED superpixels 
    spix_labels = np.array(spix_labels)                 # shape = [K_new_labelled, C+1+1]  - this array just contains the labelled superpixels and specifies the index, majority label and the average features

    # Create our empty propagation mask, ready for filling with class labels for each pixel
    prop_mask = np.empty((image_size[0], image_size[1],)) * np.nan             # shape = [H, W]

    # Now iterate again through ALL the superpixels and propagate both the known and unknown superpixels
    # for spix in np.unique(connected):
    for spix_i in range(len(np.unique(connected))):
        spix = np.unique(connected)[spix_i]
        # If the superpixel is already labelled, then propagate that label in our prop mask
        if spix in spix_labels[:,0]:
            r, c = np.where(connected == spix)  # Get indices of selected superpixel
            loc = np.where(spix_labels[:,0] == spix)
            class_label = spix_labels[loc][0][1]
            prop_mask[(r,c)] = class_label
        # If the superpixel does not have a label, we need to find the labelled superpixel with the most similiar features
        else:
            r, c = np.where(connected == spix)  # Get indices of selected superpixel
            labelled_spix_features = spix_labels[:,2:]               # shape = [K_new_labelled, C]
            one_spix_features = spix_features[spix_i,1:]              # shape = [C]
            euc_dists = [np.linalg.norm(i-one_spix_features) for i in labelled_spix_features]
            most_similiar_labelled_spix = np.argmin(np.array(euc_dists))              # shape = integer for the superpixel index with the most similiar features
            most_similiar_class_label = spix_labels[most_similiar_labelled_spix][1]    # shape = integer for corresponding class for that superpixel
            prop_mask[(r,c)] = most_similiar_class_label

    return prop_mask

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Input specifications for generating augmented ground truth from randomly distributed point labels.')

    # Paths - these are required
    parser.add_argument('-r', '-read_im', action='store', type=str, dest='read_im', help='the path to the images', required=True)
    parser.add_argument('-g', '-read_gt', action='store', type=str, dest='read_gt', help='the path to the provided labels', required=True)
    parser.add_argument('-l', '-save_labels', action='store', type=str, dest='save_labels', help='the destination of your propagated labels', required=True)
    parser.add_argument('-p', '--save_rgb', action='store', type=str, dest='save_rgb', help='the destination of your RGB propagated labels')

    # Flags to specify functionality
    parser.add_argument('--ensemble', action='store_true', dest='ensemble', help='use this flag when you would like to use an ensemble of 3 classifiers, otherwise the default is to use a single classifier')
    parser.add_argument('--points', action='store_true', dest='points', help='use this flag when your labels are already sparse, otherwise the default is dense')

    # Optional parameters
    # Default values correspond to the UCSD Mosaics dataset
    parser.add_argument('-x', '--xysigma', action='store', type=float, default=0.631, dest='xysigma', help='if NOT using ensemble and if you want to specify the sigma value for the xy component')
    parser.add_argument('-f', '--cnnsigma', action='store', type=float, default=0.5534, dest='cnnsigma', help='if NOT using ensemble and if you want to specify the sigma value for the cnn component')
    parser.add_argument('-a', '--alpha', action='store', type=float, default=1140, dest='alpha', help='if NOT using ensemble and if you want to specify the alpha value for weighting the conflict loss')
    parser.add_argument('-n', '--num_labels', action='store', type=int, default=300, dest='num_labels', help='if labels are dense, specify how many random point labels you would like to use, default is 300')
    parser.add_argument('-y', '--height', action='store', type=int, default=512, dest='image_height', help='height in pixels of images')
    parser.add_argument('-w', '--width', action='store', type=int, default=512, dest='image_width', help='width in pixels of images')
    parser.add_argument('-c', '--num_classes', action='store', type=int, default=35, dest='num_classes', help='the number of classes in the dataset')
    parser.add_argument('-u', '--unlabeled', action='store', type=int, default=34, dest='unlabeled', help='the index of the unlabeled/unknown/background class')

    args = parser.parse_args()

    read_im = args.read_im
    read_gt = args.read_gt
    save_labels = args.save_labels
    save_rgb = args.save_rgb

    ensemble = args.ensemble
    points = args.points

    sigma_xy = args.xysigma
    sigma_cnn = args.cnnsigma
    alpha = args.alpha

    num_labels = args.num_labels

    image_height = args.image_height
    image_width = args.image_width

    unlabeled = args.unlabeled

    if ensemble:
        # Feel free to change these values, these are the values we found worked well based on our ablation study
        sigma_xy_1 = 0.5597
        sigma_cnn_1 = 0.5539
        alpha_1 = 1500

        sigma_xy_2 = 0.5309
        sigma_cnn_2 = 0.846
        alpha_2 = 1590

        sigma_xy_3 = 0.631 
        sigma_cnn_3 = 0.5534
        alpha_3 = 1140
    else:
        sigma_xy = args.xysigma
        sigma_cnn = args.cnnsigma
        alpha = args.alpha

    print("received your values, setting some things up...")

    # The number of pixels used to calculated the distortion loss to increase speed and reduce memory
    num_pixels_used = 3000

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    images_done = os.listdir(save_labels)
    images = os.listdir(read_im)

    # If script is killed partway through generation, this will allow restart without repeating images
    images_filtered = [y for y in images if y not in images_done]

    NUM_CLASSES = args.num_classes

    # This is the number of superpixels
    k = 100

    # The number of superpixels along the height and width at initialization
    # Initialization is a grid, so we set to 10x10 if the image is a square
    # If the image is not a square, the superpixels should be spaced to suit
    if image_height == image_width:
        k_w = 10
        k_h = 10
    else:
        k_w = 12
        k_h = 8

    learning_rate = 0.1
    num_iterations = 50

    C = 100 # Normally this is set to 20 features
    in_channels = 5
    out_channels = 64

    norm_val_x = k_w/image_width
    norm_val_y = k_h/image_height

    # Obtain the features for the pixels in our image
    xylab_function = xylab(1.0, norm_val_x, norm_val_y)
    CNN_function = CNN(in_channels, out_channels, C) 

    model_dict = CNN_function.state_dict()
    ckp_path = "standardization_C=100_step70000.pth" # trained on UCSD, but standardization applied
    obj = torch.load(ckp_path)
    pretrained_dict = obj['net']
    # 1. filter out unnecessary keys
    # Note: in the pretrained model, all parameters have "CNN." in front of the key names, meaning they won't match the loaded CNN (when loaded without the whole SSN)
    pretrained_dict = {key[4:]: val for key, val in pretrained_dict.items() if key[4:] in model_dict}  
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict) 
    CNN_function.load_state_dict(pretrained_dict)
    CNN_function.to(device)
    CNN_function.eval()

    # Now we need to calculate the average feature (centroid) of each superpixel, based on the initialisation as a grid
    spixel_centres = get_spixel_init(k, image_width, image_height)

    # We only need to calculate metrics if we have dense ground truth
    if points == False:
        pa_metric = torchmetrics.Accuracy(num_classes = NUM_CLASSES, ignore_index=unlabeled)
        mpa_metric = torchmetrics.Accuracy(num_classes = NUM_CLASSES, ignore_index=unlabeled, average='macro')
        iou_metric = torchmetrics.JaccardIndex(num_classes = NUM_CLASSES, ignore_index=unlabeled, reduction='none')

    print("setup is complete, now iterating through your images...")

    ### Iterate through the specified images ###
    for image_name in images_filtered:
        pil_img = Image.open(os.path.join(read_im,image_name))  #.resize((image_width, image_height)
        GT_pil_img = Image.open(os.path.join(read_gt,image_name))  # .resize((image_width, image_height), Image.NEAREST

        image = np.array(pil_img)
        GT_mask_np = np.array(GT_pil_img)
        GT_mask = torch.from_numpy(GT_mask_np)
        GT_mask_torch = np.expand_dims(GT_mask, axis=2)
        transform = transforms.Compose([ToTensor()])
        GT_mask_torch = transform(GT_mask_torch)

        # If we have dense ground truth masks, we need to select num_labels pixels to propagate
        if points == False:
            # Randomly select a subset of the labelled points in the ground truth mask:
            sparse_mask = np.zeros(image_height*image_width, dtype=int)
            sparse_mask[:num_labels] = 1
            np.random.shuffle(sparse_mask)
            sparse_mask = np.reshape(sparse_mask, (image_height, image_width))
            sparse_mask = np.expand_dims(sparse_mask, axis=0)

            # We add one to all the classes so that '0' becomes all the unlabeled pixels
            sparse_labels = torch.add(GT_mask_torch, 1) * sparse_mask
            sparse_labels = torch.unsqueeze(sparse_labels, 0).to(device)                # shape = [B, 1, H, W]  

        # We are provided with randomly distributed points:
        else:
            sparse_labels = torch.unsqueeze(GT_mask_torch, 0).to(device)                # shape = [B, 1, H, W]

        means, stds = find_mean_std(image)
        image = (image - means) / stds    # shape: [H, W, C] where C is RGB in range [0,255] BUT colour channels are now standardized
        transform = transforms.Compose([img2lab(), ToTensor()])
        img_lab = transform(image)
        img_lab = torch.unsqueeze(img_lab, 0)

        image_shape = img_lab.shape                                                     # shape = [B, 3, H, W]   where 3 = RGB

        w = image_shape[3]
        h = image_shape[2]

        B = img_lab.shape[0]
        XYLab, X, Y, Lab = xylab_function(img_lab)                                     # shape = [B, 5, H, W]  where 5 = x,y,L,A,B 
        XYLab = XYLab.to(device)
        X = X.to(device)
        Y = Y.to(device)

        # send the XYLab features through the CNN to obtain the encoded features 
        with torch.no_grad():
            features = CNN_function(XYLab)                                             # shape = [B, C, H, W]  where C = 20 from config file   

        features_magnitude_mean = torch.mean(torch.norm(features, p=2, dim=1))
        features_rescaled = (features / features_magnitude_mean)
        features_cat = torch.cat((X, Y, features_rescaled), dim = 1)
        XY_cat = torch.cat((X, Y), dim = 1)
        
        mean_init = compute_init_spixel_feat(features_cat, torch.from_numpy(spixel_centres[0].flatten()).long().to(device), k)   # shape = [B, K, C]                                                                       

        CNN_features = torch.flatten(features_rescaled, start_dim=2, end_dim=3)       # shape = [B, C, N] but here we should have C = 15
        CNN_features = torch.transpose(CNN_features, 2, 1)                            # shape = [B, N, C]

        XY_features = torch.flatten(XY_cat, start_dim=2, end_dim=3)                   # shape = [B, C, N] but here we should have C = 2
        XY_features = torch.transpose(XY_features, 2, 1)                              # shape = [B, N, C]

        features_cat = torch.flatten(features_cat, start_dim=2, end_dim=3)            # shape = [B, C, N] but here we should have C = 17
        features_cat = torch.transpose(features_cat, 2, 1)                            # shape = [B, N, C]

        torch.backends.cudnn.benchmark = True
        
        if ensemble:
            criterion_1 = CustomLoss(mean_init, w*h, XY_features, CNN_features, features_cat, sparse_labels, sigma_val_xy=sigma_xy_1, sigma_val_cnn=sigma_cnn_1, alpha=alpha_1, num_pixels_used=num_pixels_used).to(device)
            optimizer_1 = Adam(criterion_1.parameters(), lr = learning_rate)
            scheduler_1 = lr_scheduler.ReduceLROnPlateau(optimizer_1, factor=0.1, patience=1, min_lr = 0.0001)

            criterion_2 = CustomLoss(mean_init, w*h, XY_features, CNN_features, features_cat, sparse_labels, sigma_val_xy=sigma_xy_2, sigma_val_cnn=sigma_cnn_2, alpha=alpha_2, num_pixels_used=num_pixels_used).to(device)
            optimizer_2 = Adam(criterion_2.parameters(), lr = learning_rate)
            scheduler_2 = lr_scheduler.ReduceLROnPlateau(optimizer_2, factor=0.1, patience=1, min_lr = 0.0001)

            criterion_3 = CustomLoss(mean_init, w*h, XY_features, CNN_features, features_cat, sparse_labels, sigma_val_xy=sigma_xy_3, sigma_val_cnn=sigma_cnn_3, alpha=alpha_3, num_pixels_used=num_pixels_used).to(device)
            optimizer_3 = Adam(criterion_3.parameters(), lr = learning_rate)
            scheduler_3 = lr_scheduler.ReduceLROnPlateau(optimizer_3, factor=0.1, patience=1, min_lr = 0.0001)

            best_clusters_1 = optimize_spix(criterion_1, optimizer_1, scheduler_1,  norm_val_x=norm_val_x, norm_val_y=norm_val_y, num_iterations=num_iterations)
            best_members_1 = members_from_clusters(sigma_xy_1, sigma_cnn_1, XY_features, CNN_features, best_clusters_1)

            best_clusters_2 = optimize_spix(criterion_2, optimizer_2, scheduler_2, norm_val_x=norm_val_x, norm_val_y=norm_val_y, num_iterations=num_iterations)
            best_members_2 = members_from_clusters(sigma_xy_2, sigma_cnn_2, XY_features, CNN_features, best_clusters_2)

            best_clusters_3 = optimize_spix(criterion_3, optimizer_3, scheduler_3, norm_val_x=norm_val_x, norm_val_y=norm_val_y, num_iterations=num_iterations)
            best_members_3 = members_from_clusters(sigma_xy_3, sigma_cnn_3, XY_features, CNN_features, best_clusters_3)

            # MAJORITY VOTE FROM THE THREE CLASSIFIERS
            best_members_1_max = torch.squeeze(torch.argmax(best_members_1, 2))
            best_members_2_max = torch.squeeze(torch.argmax(best_members_2, 2))
            best_members_3_max = torch.squeeze(torch.argmax(best_members_3, 2))

            # Clear some extra variables from the memory
            del best_members_1, best_members_2, best_members_3

            connected_1 = enforce_connectivity(best_members_1_max, h, w, k, connectivity = True)  # connectivity=True normally                       # shape = [H, W]
            connected_2 = enforce_connectivity(best_members_2_max, h, w, k, connectivity = True)  # connectivity=True normally                       # shape = [H, W]
            connected_3 = enforce_connectivity(best_members_3_max, h, w, k, connectivity = True)  # connectivity=True normally                       # shape = [H, W]

            # If there are unlabelled superpixels, we propagate the class of the superpixel with the most similar features
            prop_1 = prop_to_unlabelled_spix_feat(sparse_labels.detach().cpu(), connected_1, CNN_features, image_height, image_width)
            prop_2 = prop_to_unlabelled_spix_feat(sparse_labels.detach().cpu(), connected_2, CNN_features, image_height, image_width)
            prop_3 = prop_to_unlabelled_spix_feat(sparse_labels.detach().cpu(), connected_3, CNN_features, image_height, image_width)

            prop_1_onehot = np.eye(NUM_CLASSES, dtype=np.int32)[prop_1.astype(np.int32)]
            prop_2_onehot = np.eye(NUM_CLASSES, dtype=np.int32)[prop_2.astype(np.int32)]
            prop_3_onehot = np.eye(NUM_CLASSES, dtype=np.int32)[prop_3.astype(np.int32)]

            # Add together
            prop_count = prop_1_onehot + prop_2_onehot + prop_3_onehot

            del prop_1_onehot, prop_2_onehot, prop_3_onehot

            # The unlabeled class to be either first (0) or last
            if unlabeled == 0:
                propagated_full = np.argmax(prop_count[:,:,1:], axis=-1) + 1
                propagated_full[prop_count[:,:,0] == 3] = 0
            else:
                propagated_full = np.argmax(prop_count[:,:,:-1], axis=-1)
                propagated_full[prop_count[:,:,unlabeled] == 3] = unlabeled

        else:
            # Single classifier, so just do everything once
            criterion = CustomLoss(mean_init, w*h, XY_features, CNN_features, features_cat, sparse_labels, sigma_val_xy=sigma_xy, sigma_val_cnn=sigma_cnn, alpha=alpha, num_pixels_used=num_pixels_used).to(device)
            optimizer = Adam(criterion.parameters(), lr = learning_rate)
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=1, min_lr = 0.0001)
            best_clusters = optimize_spix(criterion, optimizer, scheduler,  norm_val_x=norm_val_x, norm_val_y=norm_val_y, num_iterations=num_iterations)
            best_members = members_from_clusters(sigma_xy, sigma_cnn, XY_features, CNN_features, best_clusters)
            connected = enforce_connectivity(torch.squeeze(torch.argmax(best_members, 2)), h, w, k, connectivity = True)  # connectivity=True normally                       # shape = [H, W]
            propagated_full = prop_to_unlabelled_spix_feat(sparse_labels.detach().cpu(), connected, CNN_features, image_height, image_width)

        # Whether using an ensemble or not, we now have a propagated mask

        # Check if the user wants us to save an RGB version of the mask and save if so
        if save_rgb is not None:
            plot_propagated(os.path.join(save_rgb, image_name[:-4]), propagated_full)

        # Save the propagated mask as a .png file in the specified directory
        propagated_as_image = Image.fromarray(propagated_full.astype(np.uint8))
        propagated_as_image.save(os.path.join(save_labels,image_name[:-4])+".png", "PNG")

        # If we started with dense ground truth, let's calculate how accurately we propagated the point labels
        if points == False:
            propagated_torch = torch.nan_to_num(torch.from_numpy(propagated_full), unlabeled).int()
            labels_torch = torch.from_numpy(GT_mask_np)

            # Find the unlabeled pixels in the original ground truth and exclude these from our metrics
            inactive_index = labels_torch == unlabeled
            propagated_torch[inactive_index] = unlabeled

            acc = pa_metric(propagated_torch, labels_torch)
            m_acc = mpa_metric(propagated_torch, labels_torch)
            m_iou = iou_metric(propagated_torch, labels_torch)

        # Clear the cache before the next image
        torch.cuda.empty_cache()

    print("propagation of point labels is complete!")

    # We can only evaluate our propagation if we have dense ground truth to compare it to
    if points == False:
        print("evaluation script working...")
        acc = pa_metric.compute()
        m_acc = mpa_metric.compute()
        miou = iou_metric.compute()

        print("per class mean intersection over union:", miou)

        class_ious_torch=miou[miou != 0]
        mean_iou_torch = torch.nanmean(class_ious_torch)

        print("PA:", acc.item()*100, ", mPA:", m_acc.item()*100, ", mIOU per class:", mean_iou_torch.item()*100)
