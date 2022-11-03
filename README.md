# PZdeepdrug

### Directory layout

    .
    ├── DeepDTNet/                   # Code for DeepDTNet
            ├── data/
                  ├── matrix/
                  ├── raw/
                  ├── embedding/
                  ├── processed/
                  └── labels/
                        ├── DCDB_deepdtnet.tsv
                        └── C_DCDB_deepdtnet.tsv
                  
            ├── create_embedding.ipynb
            ├── create_dataset.ipynb
            └── train.ipynb
    ├── MSI/                          # Code for MSI
            ├── data/
                  ├── embedding/
                  ├── processed/
                  ├── labels/
                        ├── DCDB_msi.tsv
                        └── C_DCDB_msi.tsv
            ├── create_embedding.ipynb
            ├── create_dataset.ipynb
            └── train.ipynb
    ├── NEWMIN/                       # Code for NEWMIN
            ├── data/
                  ├── embedding/
                  ├── processed/
                  ├── labels/
                        ├── DCDB_newmin.tsv
                        └── C_DCDB_newmin.tsv
            ├── create_embedding.ipynb
            ├── create_dataset.ipynb
            └── train.ipynb 
    ├── labels/                       # Original labels from DCDB, C_DCDB
            ├── DCDB_dual.tsv
            └── C_DCDB_dual.tsv
    ├── create_labels.ipynb
    └── README.md
