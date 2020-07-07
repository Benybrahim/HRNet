
Title 
==========================

## Requirement 
      
```bash
# Install dependencies
> pip install --upgrade -r requirements.txt
      
# download data
> mkdir ./data && gsutil -m cp -r {gs://.../*} ./data/
      
# download models
> mkdir ./models && gsutil -m cp -r {gs://.../*} ./models/
```
      
**NOTE**: Change global variables in `config.py`  file

## Train

#### Train all keypoints predictor model from scratch  

```bash
> cd src
> python train.py --category all --epochs 30  --batch_size 16  --gpu 0
```

#### Resume Training from a specific model.

```bash
> cd src
> python train.py --category tops --epochs 30 --batch_size 16 --resume True --resume_model {path/to/model} --init_epoch 6
```


# The directory structure
------------
```
├── README.md        
│
├── .gitignore        
│
├── requirements.txt 
│
├── models   
│   └── model.h5 
│
├── data
│   ├── images        
│   ├── train.csv    
│   └── val.csv    
│
├── source                
│   ├── config.py      
│   │                    
│   ├── train.py      
│   │  
│   ├── test.ipynb
│   │
│   └── net 
│   │   └── resnet.py
│   │
│   └── utils_ 
│       ├── generator.py
│       ├── processor.py
│       ├── evaluator.py
│       └── metrics.py
```

