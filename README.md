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
            └── train.ipynb
    ├── MSI/                          # Code for MSI
            ├── data/
                  ├── embedding/
                  ├── processed/
                  ├── labels/
                        ├── DCDB_msi.tsv
                        └── C_DCDB_msi.tsv
            ├── create_embedding.ipynb
            └── train.ipynb
    ├── NEWMIN/                       # Code for NEWMIN
            ├── data/
                  ├── embedding/
                  ├── processed/
                  ├── labels/
                        ├── DCDB_newmin.tsv
                        └── C_DCDB_newmin.tsv
            ├── create_embedding.ipynb
            └── train.ipynb 
    ├── ori_labels/                    # Original labels from DCDB, C_DCDB
            ├── DCDB_dual.tsv
            └── C_DCDB_dual.tsv
    ├── raw_database/
            ├── c_dcdb.sqlite
            ├── dcdb_components_identifier.txt
            └── dcdb.txt
    ├── create_labels_deepdtnet.ipynb            # Notebook to create labels for DeepDTNet network
    ├── create_labels_msi.ipynb                  # Notebook to create labels for MSI network
    ├── create_labels_newmin.ipynb               # Notebook to create labels for NEWMIN network
    ├── process_database.ipynb                   # Notebook to process raw_database files to ori_labels files
    └── README.md
