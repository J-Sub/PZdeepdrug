empty directory 상태이면 그 directory가 없어져서 temp.txt 파일들이 들어있으니, 파일 1개 이상 업로드 후에는 temp.txt 지워주세요!

For 정섭:  
      diffusion profile 사용하기 전에, data/raw/ directory에 10_top_msi.zip 파일을 아래 google drive link에서 다운 받아서 압축 풀어서 사용해야 함. (github 용량 문제로 github에 업로드 불가ㅠㅠ)  
      https://drive.google.com/file/d/12moZyVkUl_9bx2eCH0PbxorIXWZHEWfR/view?usp=sharing

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