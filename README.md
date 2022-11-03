# PZdeepdrug

### Directory layout
.
├── DeepDTNet/  
      ├── embedding/  
      ├── labels/
            ├── DCDB_deepdtnet.tsv
            ├── C_DCDB_deepdtnet.tsv
      ├── data/
          ├── matrix/
          ├── raw/
      ├── embedding.ipynb
      ├── create_dataset.ipynb
      ├── train.ipynb
├── MSI/
      ├── embedding/
      ├── labels/
            ├── DCDB_msi.tsv
            ├── C_DCDB_msi.tsv
      ├── data/
      ├── embedding.ipynb
      ├── create_dataset.ipynb
      ├── train.ipynb
├── labels/
      ├── DCDB_dual.tsv
      ├── C_DCDB_dual.tsv

### A typical top-level directory layout

    .
    ├── DeepDTNet/                   # Compiled files (alternatively `dist`)
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
    ├── MSI/                    # Documentation files (alternatively `doc`)
    ├── labels/                     # Source files (alternatively `lib` or `app`)
    └── README.md
