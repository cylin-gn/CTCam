
# Experimental Code for Two-Stage Fusion Model 
## for integrating GCT and vehicle flow

![RMT-dataset](https://github.com/cylin-gn/CTCam/blob/main/Figure/fusion_model.png)

This is a Pytorch implementation of the proposed 2-stage fusion model.

## Data Preparation

- The original CSV file for GCT flow is available at: [GCT flow.csv](./Data/Raw/GCT_Flow.csv)
- The original CSV file for Vehicle flow is available at: [Vehicle flow.csv](./Data/Raw/Vehicle_Flow_Raw.csv)
- To generate the **train/val/test datasets** for each type of GCT flow as {train,val,test}.npz, please follow the [script](https://github.com/liyaguang/DCRNN/blob/master/scripts/generate_training_data.py),
using the CSV files provided above.

Here is an example:

|        Date         | Road Segment 1 | ...  | Road Segment 49 | Cam1 | ... | Cam6 | 
|:-------------------:|:--------------:|:--------------:|:--------------:|:--------------:|:--------------:|:--------------:|
|         ...         |    ...         |    ...         |    ...         |    ...        |    ...        |    ...        |
| 08-28 18:55 |  81        |  ...        |   228        |    151         |    ...        |   249        |
| 08-28 19:00 |  50        |  ...        |   186        |    138         |     ...        |   205        |
| 08-29 06:00 |  20        |  ...         |   31        |    38         |    ...        |   47        |
|         ...         |    ...         |    ...         |    ...         |      ...        |   ...        |

### train/test/val dataset

The train/test/val data is now provided at : 
```
../Data
```

#### How to Create

We split data in 7:2:1 for generating train/test/val data.

Run the [scripts](https://github.com/liyaguang/DCRNN/blob/master/scripts/generate_training_data.py) to generate the train/test/val dataset.

or, see the "Dataset Zoo" below.

## Graph Construction
As the implementation is based on pre-calculated distances between road sections, we provided the CSV file with road section distances and IDs at: 
- GCT Flow: [Road Section Distance](https://github.com/cylin-gn/CTCam/blob/main/Data/Raw/GCT_Roads_Distance.txt). 
- Vehicle Flow: [Road Section Distance](https://github.com/cylin-gn/CTCam/blob/main/Data/Raw/Vehicle_Flow_Roads_Distance.txt). 

Run the [script](https://github.com/liyaguang/DCRNN/blob/master/scripts/gen_adj_mx.py) to generate the Graph Structure based on the "Road Section Distance" file provided above.

The `processed Graph Structure of Road Section Network` is available at: 
- GCT Flow: [road network structure file](https://github.com/cylin-gn/CTCam/blob/main/Data/hsin_49_GCT_0600_1900_rename/adj_mat_49_rename.pkl)
- Vehicle Flow: [road network structure file](https://github.com/cylin-gn/CTCam/blob/main/Data/hsin_6_CCTV_0600_1900_rename/adj_mat_6_rename.pkl)


## Dataset Zoo
The **processed train/val/test data structures file** is available, 
- data structures file for GCT flow: [Here](https://github.com/cylin-gn/CTCam/tree/main/Data/hsin_49_GCT_0600_1900_rename)
- data structures file for Vehiclw flow: [Here](https://github.com/cylin-gn/CTCam/tree/main/Data/hsin_6_CCTV_0600_1900_rename)

## Model Zoo

Download and store the trained models in 'pretrained' folder as follow:

```
./Model/save
```
This is the example that using [Graph Wavenet](https://github.com/nnzhan/Graph-WaveNet) as extractor and third-GCN-based model as shown in paper:
- Pretrained Extractor for GCT Flow: [exp202306161737_0.pth](./Model/save/exp202306161737_0.pth)
- Pretrained Extractor for Vehicle Flow: [exp202306161734_0.pth](./Model/save/exp202306161734_0.pth)
- Fusion & Third-GCN-based Training: [exp202306162115_0.pth](./Model/save/exp202306162115_0.pth)


## Model Training

- Training Extractors for GCT flow prediction and Vehicle flow prediction, similar to traffic speed prediction.
  For using Graph Wavenet as an example, please follow the instruction in [Graph Wavenet](https://github.com/nnzhan/Graph-WaveNet) 
- Put the pre-trained-well extractors model in:
```
./Model/save
```
- Please set the location of the dataset and graph structure file in `argparse.ArgumentParser()` of `parameters.py`

For GCT Flow extractors, please set in the:
```
### GCT ###
...
```

For Vehicle Flow extractors, please set in the:
```
### CCTV ###
...
```

For Stage Two, please set in the:
```
### Fusion ###
...
```

And put all codes together to run the training process.

Or directly run the `Jupyter Notebook`:

```
2_stages_fusion.ipynb
```

for Stage Two training with our provided pre-trained extractors.
