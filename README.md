# PZdeepdrug

### Directory layout

    .
    ├── DeepDTNet/                   # Code for DeepDTNet
            ├── embedding/
            ├── labels/
                  ├── DCDB_deepdtnet.tsv
                  └── C_DCDB_deepdtnet.tsv
            ├── data/
                  ├── matrix/
                  └── raw/
            ├── embedding.ipynb
            ├── create_dataset.ipynb
            └── train.ipynb
    ├── MSI/                          # Code for MSI
            ├── embedding/
            ├── labels/
                  ├── DCDB_msi.tsv
                  └── C_DCDB_msi.tsv
            ├── data/
            ├── embedding.ipynb
            ├── create_dataset.ipynb
            └── train.ipynb
    ├── labels/                       # Original labels from DCDB, C_DCDB
            ├── DCDB_dual.tsv
            └── C_DCDB_dual.tsv
    └── README.md
