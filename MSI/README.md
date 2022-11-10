empty directory 상태이면 그 directory가 없어져서 temp.txt 파일들이 들어있으니, 파일 1개 이상 업로드 후에는 temp.txt 지워주세요!

### Directory layout
    .
    ├── MSI/                          # Code for MSI
            ├── data/
                  ├── embedding/
                  ├── processed/
                  ├── raw/
                  ├── labels/
                        ├── DCDB_msi.tsv
                        └── C_DCDB_msi.tsv
            ├── load_data.py
            ├── create_embedding.ipynb
            ├── train.ipynb
            └── README.md