CONFIG = {
    "version": {
        "dataset_Version": "v1",
        "input_data_path" :"/Users/niketanand/Documents/MLOps/DVC_CI_CD/CIFAR-10-images-master/",
        "output_data" : "/Users/niketanand/Documents/MLOps/DVC_CI_CD/CIFAR-10-images-master/",
        "seed" : {400}
    },

    "split_ratio": {
        "train":"{80}",
        "test" : "{10}",
        "val" : "{10}",  
    },

    "parameter":{
        'lr':'{0}',
        'batch_size':'{0}',
        'num_epochs':'{0}',
        'conv_layers':'{32},{64},{128},{256}',
        'conv_filters':'{3}',
        'kernel_sizes':'{0}',
    }

}