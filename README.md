# Licenta2020ETTI Vlad-Andrei Radu

Pentru a rula script-urile este necesar setul de date **endomondoHR_proper.json**, diponibil la adresa https://sites.google.com/eng.ucsd.edu/fitrec-project/home

## Script-uri:

### LSTM_GRU_GPU.py
Aici sunt cele două rețele neurale dezvoltate de mine. Tot aici se face majoritatea preprocesării, antrenarea și testarea lor.

### LoadDB.py
Cu acest script se încarcă datele din fișierul **endomondoHR_proper.json** și se elimină utilizatorii care nu au suficiente antrenamente.

### data_split.py
Acest script creează fișierul **endomondoHR_proper_temporal_dataset.pkl**. Este preluat de la autorii FitRec și adaptat pentru a rula pe Windows.

### heart_rate_aux_LSTM.py
Rețeaua FitRec cu celule LSTM. Am adaptat codul pentru a putea rula pe Windows.

### heart_rate_aux_GRU.py
Rețeaua FitRec modificată pentru a utiliza celule GRU.

### data_interpreter_Keras_aux.py
Am adaptat fișierul pentru a putea rula pe Windows. Funcțiile de aici sunt apelate în *heart_rate_aux_LSTM.py* și *heart_rate_aux_GRU.py*
