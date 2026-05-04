import os
import torch
from dataclasses import dataclass
from torch.utils.data import DataLoader

from sample4geo.dataset.university import U1652DatasetEval, get_transforms
from sample4geo.evaluate.university import evaluate
from sample4geo.model import TimmModel


@dataclass
class Configuration:

    model: str = 'vit_base_patch14_dinov2.lvd142m'
    
    # Override model image size
    img_size: int = 518
    
    # Evaluation
    batch_size: int = 32
    verbose: bool = True
    gpu_ids: tuple = (0,1)
    normalize_features: bool = True
    eval_gallery_n: int = -1             # -1 for all or int
    
    # Dataset
    dataset: str = 'U1652-S2D'           # 'U1652-D2S' | 'U1652-S2D'
    data_folder: str = "./data/U1652"
    
    # Checkpoint to start from
    checkpoint_start = '/public/home/houkaiji/flower/university/vit_base_patch14_dinov2.lvd142m/015033/weights_e10_0.8454.pth'
  
    # set num_workers to 0 if on Windows
    num_workers: int = 0 
    
    # train on GPU if available
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu' 
    

#-----------------------------------------------------------------------------#
# Config                                                                      #
#-----------------------------------------------------------------------------#

config = Configuration() 

if config.dataset == 'U1652-D2S':
    config.query_folder_train = './data/U1652/train/satellite'
    config.gallery_folder_train = './data/U1652/train/drone'   
    # config.query_folder_test = '/public/home/chuyanhao/Sample4Geo/data/test/query_drone' 
    config.query_folder_test = f'/public/home/houkaiji/Sample4Geo/data/query_drone_mirror20'
    config.gallery_folder_test = f'/public/home/houkaiji/dataset/U1652/University-Release/test/gallery_satellite'   
elif config.dataset == 'U1652-S2D':
    config.query_folder_train = f'/public/home/houkaiji/dataset/U1652/University-Release/train/satellite'
    config.gallery_folder_train = f'/public/home/houkaiji/dataset/U1652/University-Release/train/drone'
    config.query_folder_test = f'/public/home/houkaiji/dataset/U1652/University-Release/test/query_satellite'
    config.gallery_folder_test = f'/public/home/houkaiji/dataset/U1652/University-Release/test/gallery_drone'


if __name__ == '__main__':

    #-----------------------------------------------------------------------------#
    # Model                                                                       #
    #-----------------------------------------------------------------------------#
        
    print("\nModel: {}".format(config.model))


    model = TimmModel(config.model,
                          pretrained=True,
                          img_size=config.img_size)
                          
    data_config = model.get_config()
    # print(data_config)
    mean = data_config["mean"]
    std = data_config["std"]
    img_size = (config.img_size, config.img_size)
    

    # load pretrained Checkpoint    
    if config.checkpoint_start is not None:  
        print("Start from:", config.checkpoint_start)
        model_state_dict = torch.load(config.checkpoint_start)  
        model.load_state_dict(model_state_dict, strict=False)     

    # Data parallel
    print("GPUs available:", torch.cuda.device_count())  
    if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=config.gpu_ids)
            
    # Model to device   
    model = model.to(config.device)
    # print(model)
    print("\nImage Size Query:", img_size)
    print("Image Size Ground:", img_size)
    print("Mean: {}".format(mean))
    print("Std:  {}\n".format(std)) 


    #-----------------------------------------------------------------------------#
    # DataLoader                                                                  #
    #-----------------------------------------------------------------------------#

    # Transforms
    val_transforms, train_sat_transforms, train_drone_transforms = get_transforms(img_size, mean=mean, std=std)
                                                                                                                                 
    
    # Reference Satellite Images
    query_dataset_test = U1652DatasetEval(data_folder=config.query_folder_test,
                                               mode="query",
                                               transforms=val_transforms,
                                               )
    
    query_dataloader_test = DataLoader(query_dataset_test,
                                       batch_size=config.batch_size,
                                       num_workers=config.num_workers,
                                       shuffle=False,
                                       pin_memory=True)
    
    # Query Ground Images Test
    gallery_dataset_test = U1652DatasetEval(data_folder=config.gallery_folder_test,
                                               mode="gallery",
                                               transforms=val_transforms,
                                               sample_ids=query_dataset_test.get_sample_ids(),
                                               gallery_n=config.eval_gallery_n,
                                               )
    
    gallery_dataloader_test = DataLoader(gallery_dataset_test,
                                       batch_size=config.batch_size,
                                       num_workers=config.num_workers,
                                       shuffle=False,
                                       pin_memory=True)
    
    print("Query Images Test:", len(query_dataset_test))
    print("Gallery Images Test:", len(gallery_dataset_test))
   

    print("\n{}[{}]{}".format(30*"-", "University-1652", 30*"-"))  

    r1_test,index = evaluate(config=config,
                       model=model,
                       query_loader=query_dataloader_test,
                       gallery_loader=gallery_dataloader_test, 
                       ranks=[1, 5, 10],
                       step_size=1000,
                       cleanup=True)
    print("index:",gallery_dataset_test.sample_ids[int(index)])
    
 
