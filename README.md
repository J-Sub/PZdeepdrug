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
            ├── train.ipynb
            └── README.md
    ├── MSI/                          # Code for MSI
            ├── data/
                  ├── embedding/
                  ├── processed/
                  ├── labels/
                        ├── DCDB_msi.tsv
                        └── C_DCDB_msi.tsv
            ├── create_embedding.ipynb
            ├── train.ipynb
            └── README.md
    ├── NEWMIN/                       # Code for NEWMIN
            ├── data/
                  ├── embedding/
                  ├── processed/
                  ├── labels/
                        ├── DCDB_newmin.tsv
                        └── C_DCDB_newmin.tsv
            ├── create_embedding.ipynb
            ├── train.ipynb
            └── README.md
    ├── ori_labels/                    # Original labels from DCDB, C_DCDB
            ├── DCDB_dual.tsv
            └── C_DCDB_dual.tsv
    ├── raw_database/
            ├── c_dcdb.sqlite
            ├── dcdb_components_identifier.txt
            ├── dcdb.txt
            └── README.md
    ├── create_labels_deepdtnet.ipynb            # Notebook to create labels for DeepDTNet network
    ├── create_labels_msi.ipynb                  # Notebook to create labels for MSI network
    ├── create_labels_newmin.ipynb               # Notebook to create labels for NEWMIN network
    ├── process_database.ipynb                   # Notebook to process raw_database files to ori_labels files
    └── README.md
