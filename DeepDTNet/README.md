### Directory layout

    .
    ├── DeepDTNet/                   # Code for DeepDTNet
            ├── data/
                  ├── matrix/                      # PPMI matrix files
                  ├── raw/                         # DeepDTNet에서 사용하는 network, id mapping dictionary 등의 files
                  ├── embedding/                   # create_embedding.ipynb에서 생성된 drug embedding vector files
                  ├── processed/                   # train.ipynb에서 생성된 dataset을 저장
                  └── labels/                      # DeepDTNet의 drug들의 combination label
                        ├── DCDB_deepdtnet.tsv
                        └── C_DCDB_deepdtnet.tsv
            ├── SDAE.py                            #
            ├── create_embedding.ipynb             # drug embedding vector 만드는 notebook file
            └── train.ipynb                        # dataset을 구성하고 학습하는 notebook file

~~~python
# Steps for creating drug embedding vector
1. create_embedding.ipynb
2. run SDAE.py
   
# Example for run SDAE.py   
python SDAE.py --matrix_dir ./data/matrix/PPMI_matrix_BiologicalProcess.txt --hidden_size 12
python SDAE.py --matrix_dir ./data/matrix/PPMI_matrix_BiologicalProcess.txt --hidden_size 3
~~~