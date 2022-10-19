#!/bin/sh
python SDAE.py --matrix_dir ./matrix/PPMI_matrix_BiologicalProcess.txt --hidden_size 12 && echo "completed with python_file.py" &
python SDAE.py --matrix_dir ./matrix/PPMI_matrix_CellularComponent.txt --hidden_size 12 && echo "completed with python_file.py" &
python SDAE.py --matrix_dir ./matrix/PPMI_matrix_Chemical.txt --hidden_size 12 && echo "completed with python_file.py" &
python SDAE.py --matrix_dir ./matrix/PPMI_matrix_MolecularFunction.txt --hidden_size 12 && echo "completed with python_file.py" &
python SDAE.py --matrix_dir ./matrix/PPMI_matrix_ProteinSequence.txt --hidden_size 12 && echo "completed with python_file.py" &
python SDAE.py --matrix_dir ./matrix/PPMI_matrix_Therapeutic.txt --hidden_size 12 && echo "completed with python_file.py" &

python SDAE.py --matrix_dir ./matrix/PPMI_matrix_BiologicalProcess.txt --hidden_size 3 && echo "completed with python_file.py" &
python SDAE.py --matrix_dir ./matrix/PPMI_matrix_CellularComponent.txt --hidden_size 3 && echo "completed with python_file.py" &
python SDAE.py --matrix_dir ./matrix/PPMI_matrix_Chemical.txt --hidden_size 3 && echo "completed with python_file.py" &
python SDAE.py --matrix_dir ./matrix/PPMI_matrix_MolecularFunction.txt --hidden_size 3 && echo "completed with python_file.py" &
python SDAE.py --matrix_dir ./matrix/PPMI_matrix_ProteinSequence.txt --hidden_size 3 && echo "completed with python_file.py" &
python SDAE.py --matrix_dir ./matrix/PPMI_matrix_Therapeutic.txt --hidden_size 3 && echo "completed with python_file.py" &


