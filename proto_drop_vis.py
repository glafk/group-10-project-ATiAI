import random
from util.vis_pipnet import * 
from main import *

from pipnet.pipnet import PIPNet, get_network
from util.args import get_args, save_args, get_optimizer_nn

import argparse
import torch
import torch.nn as nn

@torch.no_grad()                    
def visualize_topk_dropped(proto_dropped, proto_kept, log_dir, path_trained_model, foldername, k=10, to_sample=20, num_features=768):
    
    args = argparse.Namespace(dataset='plankton', validation_size=0.0, net='convnext_tiny_26', batch_size=16, 
              batch_size_pretrain=128, epochs=0, epochs_pretrain=0, optimizer='Adam', lr=0.05, lr_block=0.0005, 
              lr_net=0.0005, weight_decay=0.0, disable_cuda=False, log_dir=log_dir, 
              num_features=num_features, image_size=128, state_dict_dir_net=path_trained_model, 
              freeze_epochs=0, dir_for_saving_images='visualization_results', 
              disable_pretrained=False, weighted_loss=False, seed=1, gpu_ids='', num_workers=8, 
              bias=False, extra_test_image_folder='./experiments', wshape=14)
    
    device_ids = []
    if args.gpu_ids!='':
        for m in range(len(gpu_list)):
            device_ids.append(int(gpu_list[m]))

    global device
    if not args.disable_cuda and torch.cuda.is_available():
        if len(device_ids)==1:
            device = torch.device('cuda:{}'.format(args.gpu_ids))
        elif len(device_ids)==0:
            device = torch.device('cuda')
            print("CUDA device set without id specification", flush=True)
            device_ids.append(torch.cuda.current_device())
        else:
            print("This code should work with multiple GPU's but we didn't test that, so we recommend to use only 1 GPU.", flush=True)
            device_str = ''
            for d in device_ids:
                device_str+=str(d)
                device_str+=","
            device = torch.device('cuda:'+str(device_ids[0]))
    else:
        device = torch.device('cpu')

    # Log which device was actually used
    print("Device used: ", device, "with id", device_ids, flush=True)
    
    feature_net, add_on_layers, pool_layer, classification_layer, num_prototypes = get_network(86, args)
   
    # Create a PIP-Net
    net = PIPNet(num_classes=86,
                    num_prototypes=num_prototypes,
                    feature_net = feature_net,
                    args = args,
                    add_on_layers = add_on_layers,
                    pool_layer = pool_layer,
                    classification_layer = classification_layer
                    )
    net = net.to(device=device)
    net = nn.DataParallel(net, device_ids = device_ids)  
    optimizer_net, optimizer_classifier, params_to_freeze, params_to_train, params_backbone = get_optimizer_nn(net, args)  

    with torch.no_grad():
        epoch = 0
        checkpoint = torch.load(args.state_dict_dir_net,map_location=device)
        net.load_state_dict(checkpoint['model_state_dict'],strict=True) 
        print("Pretrained network loaded", flush=True)
        net.module._multiplier.requires_grad = False

        
    trainloader, trainloader_pretraining, trainloader_normal, trainloader_normal_augment, projectloader, testloader, test_projectloader, classes = get_dataloaders(args, device)
    
    
    print("Visualizing prototypes for topk...", flush=True)
    dir = os.path.join(args.log_dir, foldername)
    if not os.path.exists(dir):
        os.makedirs(dir)

    near_imgs_dirs = dict()
    seen_max = dict()
    saved = dict()
    saved_ys = dict()
    tensors_per_prototype = dict()
    
    for p in range(net.module._num_prototypes):
        near_imgs_dir = os.path.join(dir, str(p))
        near_imgs_dirs[p]=near_imgs_dir
        seen_max[p]=0.
        saved[p]=0
        saved_ys[p]=[]
        tensors_per_prototype[p]=[]
    
    patchsize, skip = get_patch_size(args)

    imgs = projectloader.dataset.imgs
    
    # Make sure the model is in evaluation mode
    net.eval()
    classification_weights = net.module._classification.weight

    # Show progress on progress bar
    img_iter = tqdm(enumerate(projectloader),
                    total=len(projectloader),
                    mininterval=50.,
                    desc='Collecting topk',
                    ncols=0)

    # Iterate through the data
    images_seen = 0
    topks = dict()
    # Iterate through the training set
    for i, (xs, ys) in img_iter:
        images_seen+=1
        xs, ys = xs.to(device), ys.to(device)

        with torch.no_grad():
            # Use the model to classify this batch of input data
            pfs, pooled, _ = net(xs, inference=True)
            pooled = pooled.squeeze(0) 
            pfs = pfs.squeeze(0) 
            
            for p in range(pooled.shape[0]):
                if p not in topks.keys():
                    topks[p] = []

                if len(topks[p]) < k:
                    topks[p].append((i, pooled[p].item()))
                else:
                    topks[p] = sorted(topks[p], key=lambda tup: tup[1], reverse=True)
                    if topks[p][-1][1] < pooled[p].item():
                        topks[p][-1] = (i, pooled[p].item())
                    if topks[p][-1][1] == pooled[p].item():
                        # equal scores. randomly chose one (since dataset is not shuffled so latter images with same scores can now also get in topk).
                        replace_choice = random.choice([0, 1])
                        if replace_choice > 0:
                            topks[p][-1] = (i, pooled[p].item())

    alli = []
    prototypes_not_used = []
    for p in topks.keys():
        found = False
        for idx, score in topks[p]:
            alli.append(idx)
            if score > 0.1:  #in case prototypes have fewer than k well-related patches
                found = True
#         if not found:
#             prototypes_not_used.append(p)

    print(len(prototypes_not_used), "prototypes do not have any similarity score > 0.1. Will be ignored in visualisation.")
    abstained = 0
    # Show progress on progress bar
    img_iter = tqdm(enumerate(projectloader),
                    total=len(projectloader),
                    mininterval=50.,
                    desc='Visualizing topk',
                    ncols=0)
    for i, (xs, ys) in img_iter: #shuffle is false so should lead to same order as in imgs
        if i in alli:
            xs, ys = xs.to(device), ys.to(device)
            for p in topks.keys():
                if p not in prototypes_not_used:
                    for idx, score in topks[p]:
                        if idx == i:
                            # Use the model to classify this batch of input data
                            with torch.no_grad():
                                softmaxes, pooled, out = net(xs, inference=True) #softmaxes has shape (1, num_prototypes, W, H)
                                outmax = torch.amax(out,dim=1)[0] #shape ([1]) because batch size of projectloader is 1
                                if outmax.item() == 0.:
                                    abstained+=1
                            
                            # Take the max per prototype.                             
                            max_per_prototype, max_idx_per_prototype = torch.max(softmaxes, dim=0)
                            max_per_prototype_h, max_idx_per_prototype_h = torch.max(max_per_prototype, dim=1)
                            max_per_prototype_w, max_idx_per_prototype_w = torch.max(max_per_prototype_h, dim=1) #shape (num_prototypes)
                            
                            c_weight = torch.max(classification_weights[:,p]) #ignore prototypes that are not relevant to any class
                                
                            h_idx = max_idx_per_prototype_h[p, max_idx_per_prototype_w[p]]
                            w_idx = max_idx_per_prototype_w[p]

                            img_to_open = imgs[i]
                            if isinstance(img_to_open, tuple) or isinstance(img_to_open, list): #dataset contains tuples of (img,label)
                                img_to_open = img_to_open[0]

                            image = transforms.Resize(size=(args.image_size, args.image_size))(Image.open(img_to_open))
                            img_tensor = transforms.ToTensor()(image).unsqueeze_(0) #shape (1, 3, h, w)
                            h_coor_min, h_coor_max, w_coor_min, w_coor_max = get_img_coordinates(args.image_size, softmaxes.shape, patchsize, skip, h_idx, w_idx)
                            img_tensor_patch = img_tensor[0, :, h_coor_min:h_coor_max, w_coor_min:w_coor_max]

                            saved[p]+=1
                            tensors_per_prototype[p].append(img_tensor_patch)

    print("Abstained: ", abstained, flush=True)
    all_proto_dropped = []
    all_proto_kept = []
    
    for p in range(net.module._num_prototypes):
        if saved[p]>0:
             # add text next to each topk-grid, to easily see which prototype it is
            text = "P "+str(p)
            txtimage = Image.new("RGB", (img_tensor_patch.shape[1],img_tensor_patch.shape[2]), (0, 0, 0))
            draw = D.Draw(txtimage)
            draw.text((img_tensor_patch.shape[0]//2, img_tensor_patch.shape[1]//2), text, anchor='mm', fill="white")
            txttensor = transforms.ToTensor()(txtimage)
            tensors_per_prototype[p].append(txttensor)
            # save top-k image patches in grid
            try:
                grid = torchvision.utils.make_grid(tensors_per_prototype[p], nrow=k+1, padding=1)
                torchvision.utils.save_image(grid,os.path.join(dir,"grid_topk_%s.png"%(str(p))))
                
                if saved[p]>=k:
                    if p in proto_kept:
                        all_proto_kept.append(tensors_per_prototype[p])
                    if p in proto_dropped:
                        all_proto_dropped.append(tensors_per_prototype[p])
            except:
                pass
            
    all_proto_kept_sample = [item for sublist in random.sample(all_proto_kept, to_sample) for item in sublist] 
    
    grid = torchvision.utils.make_grid(all_proto_kept_sample, nrow=k+1, padding=1)
    torchvision.utils.save_image(grid,os.path.join(dir,"grid_topk_all_proto_kept_sample.png"))
    
    all_proto_dropped_sample = [item for sublist in random.sample(all_proto_dropped, to_sample) for item in sublist] 
    
    grid = torchvision.utils.make_grid(all_proto_dropped_sample, nrow=k+1, padding=1)
    torchvision.utils.save_image(grid,os.path.join(dir,"grid_topk_all_proto_dropped_sample.png"))
        
    all_proto_kept = [item for sublist in all_proto_kept for item in sublist]
    
    all_proto_dropped = [item for sublist in all_proto_dropped for item in sublist]

    grid = torchvision.utils.make_grid(all_proto_kept, nrow=k+1, padding=1)
    torchvision.utils.save_image(grid,os.path.join(dir,"grid_topk_all_proto_kept.png"))
    
        
    grid = torchvision.utils.make_grid(all_proto_dropped, nrow=k+1, padding=1)
    torchvision.utils.save_image(grid,os.path.join(dir,"grid_topk_all_dropped.png"))
    
    return 

