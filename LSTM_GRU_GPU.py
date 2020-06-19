import os
os.environ['HDF5_DISABLE_VERSION_CHECK'] = '2'  #Pentru a ascunde avertismentele legate de HDF5
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'        #pentru a ascunde avertismentele/logurile TF.

import tensorflow as tf
from LoadDB import readlines, trim
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, GRU
from keras import backend as K
import matplotlib.pyplot as plt
import time
from haversine import distance as hav_dist
from scipy import interpolate

def variables(dataset):
    # 3 variabile in: altitude, speed, distance (doar pt cei care au si speed).
    # 1 var. out: heart_rate
    heart_rate = []
    speed = []
    altitude = []
    distance = []
    longitude = []
    latitude = []
    timestamps = []
    for i in range(0, len(dataset)):
        if 'speed' in dataset[i]:
            heart_rate.append(dataset[i]['heart_rate'])
            speed.append(dataset[i]['speed'])
            altitude.append(dataset[i]['altitude'])
            ### Pt calculul distantei
            longitude.append(dataset[i]['longitude'])
            latitude.append(dataset[i]['latitude'])
            timestamps.append(dataset[i]['timestamp'])

    # calculul distantei pt fiecare antrenament (i) si fiecare timestep(j)
    for i in range(0, len(latitude)):
        dist_temp = []
        dist_temp.append(0)
        # pornim de la 1 pt ca initializam prima dist. cu 0:
        for j in range(1, len(latitude[i])):
            orig = (latitude[i][j - 1], longitude[i][j - 1])
            dest = (latitude[i][j], longitude[i][j])
            dist_temp.append(hav_dist(orig, dest))
        distance.append(dist_temp)

    altitude = np.array(altitude)
    speed = np.array(speed)
    heart_rate = np.array(heart_rate)
    distance = np.array(distance)
    timestamps = np.array(timestamps)

    return altitude, speed, distance, heart_rate, timestamps

def interpolare(y, x):
    fct = interpolate.splrep(x,y) #se creeaza curba de interpolare
    new_x = np.arange(x[0],x[0]+2000, 4) # pastram 2000 de secunde / 4 secunde - 500 mom timp
    return interpolate.splev(x, fct) #se iau noile valori in functie de curba

def preprocess(altitude, speed, distance, heart_rate):
    # transformam 3D -> [samples, timesteps, features]
    l = altitude.shape[0]   #prima dimensiune a matricii - numarul de antrenamente
    c = altitude.shape[1]   #a doua dimensiune a matricii - numarul de esantioane de timp
    a = 3                   #a 3-a dimensiune a matricii - numarul de variabile de intrare
    dataInput = np.empty((l, c, a))
    for i in range(altitude.shape[0]):
        for j in range(altitude.shape[1]):
            dataInput[i,j,0] = altitude[i,j]
            dataInput[i,j,1] = speed[i,j]
            dataInput[i,j,2] = distance[i][j]
    dataOutput = heart_rate

    return dataInput, dataOutput

def normalizare(altitude, speed, distance, heart_rate):
    # normalizarea altitudinii
    scaler_alt = MinMaxScaler(feature_range=(0, 1))
    scaled_alt = scaler_alt.fit_transform(altitude)
    scaled_alt = np.array(scaled_alt)
    altitude = scaled_alt

    # normalizarea vitezei
    scaler_spd = MinMaxScaler(feature_range=(0, 1))
    scaled_spd = scaler_spd.fit_transform(speed)
    scaled_spd = np.array(scaled_spd)
    speed = scaled_spd

    # normalizarea distantei
    scaler_dst = MinMaxScaler(feature_range=(0, 1))
    scaled_dst = scaler_dst.fit_transform(distance)
    scaled_dst = np.array(scaled_dst)
    distance = scaled_dst

    # normalizarea pulsului
    scaler_hr = MinMaxScaler(feature_range=(0, 1))
    scaled_hr = scaler_hr.fit_transform(heart_rate)
    scaled_hr = np.array(scaled_hr)
    heart_rate = scaled_hr

    # returnam si scaler_hr pentru denormalizare mai tarziu
    return altitude, speed, distance, heart_rate, scaler_hr

def main():
    batch_sz = 128
    learning_rate = 0.005
    dataset_size = 100000
    min_workouts = 20
    max_workouts = 500
    epochs_num = 50
    cell_size = 150
    workout = 15
    val_loss_patience = 5 #Numarul maxim de epoci dupa care daca nu mai scade val_loss ne oprim.
    prag_puls_high = 163.8
    prag_puls_low =144

    # Path-uri
    graph_path = 'D:/OneDrive/Education/College/Licenta/RecSys - Current/FitRecStuff/LSTM_vs_GRU/'
    figure_string = str(batch_sz) + "_" + str(dataset_size) + "_" + str(min_workouts) + "_" + str(
        max_workouts) + "_" + str(epochs_num) + "_" + str(cell_size) + "_" + str(workout)
    file_path = open((graph_path + "log_" + figure_string + ".txt"), 'w')

    dataset = readlines(size=dataset_size)
    dataset = trim(dataset, minWorkouts=min_workouts, maxWorkouts=max_workouts)
    [altitude, speed, distance, heart_rate, timestamps] = variables(dataset)


    #eliminarea antrenamentelor pt care esantionarea nu e cronologica...
    chrono = np.where(np.diff(timestamps) < 0)[0]
    altitude = np.delete(altitude, chrono, axis = 0)
    speed = np.delete(speed, chrono, axis = 0)
    distance = np.delete(distance, chrono, axis = 0)
    heart_rate = np.delete(heart_rate, chrono, axis = 0)
    timestamps = np.delete(timestamps, chrono, axis = 0)

    #interpolare
    for i in range(timestamps.shape[0]):
        altitude[i] = interpolare(altitude[i],timestamps[i])
        speed[i] = interpolare(speed[i],timestamps[i])
        distance[i] = interpolare(distance[i], timestamps[i])
        heart_rate[i] = interpolare(heart_rate[i],timestamps[i])
    #eliminarea antrenamentelor pentru care nu s-a putut face interpolarea (de ex. cele prea scurte)
    remove_rows = []
    for i in range(timestamps.shape[0]):    #se noteaza toate antrenamentele care contin valori nan.
        if np.isnan(altitude[i]).any() or np.isnan(speed[i]).any() or \
            np.isnan(distance[i]).any() or np.isnan(heart_rate[i]).any():
            remove_rows.append(i)
    altitude = np.delete(altitude,remove_rows, axis = 0)
    speed = np.delete(speed,remove_rows, axis = 0)
    distance = np.delete(distance,remove_rows, axis = 0)
    heart_rate = np.delete(heart_rate,remove_rows, axis = 0)



    # calculul timpului mediu intre esantioane
    dif = np.empty(shape =(timestamps.shape[0],timestamps.shape[1]))
    for i in range(timestamps.shape[0]):
        for j in range(1,timestamps.shape[1]):
            dif[i][j] = timestamps[i][j]-timestamps[i][j-1]

    #normalizarea datelor
    altitude, speed, distance, heart_rate, scaler_hr = normalizare(altitude, speed, distance, heart_rate)

    #10% test, 15% validare 75% antrenare

    altitude_train, altitude_test, speed_train, speed_test, distance_train, distance_test, heart_rate_train, heart_rate_test = \
        train_test_split(altitude, speed, distance, heart_rate, test_size=0.25, random_state=0)
    # test_size = 0.4 pt ca vrem ca 10% din total sa fie de test.
    altitude_valid, altitude_test, speed_valid, speed_test, distance_valid, distance_test, heart_rate_valid, heart_rate_test = \
        train_test_split(altitude_test,speed_test,distance_test,heart_rate_test, test_size = 0.4, random_state= 0)

    dataInput_train, dataOutput_train = preprocess(altitude_train, speed_train, distance_train, heart_rate_train)
    dataInput_valid, dataOutput_valid = preprocess(altitude_valid, speed_valid, distance_valid, heart_rate_valid)
    dataInput_test, dataOutput_test = preprocess(altitude_test, speed_test, distance_test, heart_rate_test)


    with tf.device('/gpu:0'):
        # monitorizam modificarea erorii pe setul de validare si ne oprim dupa val_loss_patience epoci in care nu mai scade
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience = val_loss_patience, restore_best_weights= True)

        # definim reteaua LSTM
        LSTMmodel = Sequential()
        LSTMmodel.add(LSTM(cell_size, input_shape=(dataInput_train.shape[1], dataInput_train.shape[2])))
        LSTMmodel.add(Dense(500))
        LSTMmodel.compile(loss='mse', optimizer='adam')
        # Dupa compilarea modelului ii modificam rata de invatare
        K.set_value(LSTMmodel.optimizer.learning_rate, learning_rate)
        #LSTMmodel.summary()

        # definim reteaua GRU
        GRUmodel = Sequential()
        GRUmodel.add(GRU(cell_size, input_shape=(dataInput_train.shape[1], dataInput_train.shape[2])))
        GRUmodel.add(Dense(500))
        GRUmodel.compile(loss='mse', optimizer='adam')
        K.set_value(GRUmodel.optimizer.learning_rate, learning_rate)
        #GRUmodel.summary()

    with tf.device('/gpu:0'):
        # fit GRU network - antrenare GRU
        gru_start = time.time()
        # """
        GRUhistory = GRUmodel.fit(dataInput_train, dataOutput_train, epochs=epochs_num, batch_size=batch_sz,
                                  validation_data=(dataInput_valid, dataOutput_valid), verbose=1, callbacks=[callback],
                                  shuffle=False)
        """
        # daca dorim incarcarea unui model preantrenat in loc
        GRUmodel.load_weights(graph_path + "model_checkpoints/GRU_"+figure_string)
        """
        gru_end = time.time()
        gru_dur = gru_end - gru_start

    with tf.device('/gpu:0'):
        # fit LSTM network - antrenare LSTM
        lstm_start = time.time()
        # """
        LSTMhistory = LSTMmodel.fit(dataInput_train, dataOutput_train, epochs=31, batch_size=batch_sz,
                                    validation_data=(dataInput_valid, dataOutput_valid), verbose=1,
                                    #callbacks=[callback],
                                    shuffle=False)  # verbose = 2 afiseaza doar progresul pe epoci. verbose = 1 arata si detalii
        """
        # daca dorim incarcarea unui model preantrenat
        LSTMmodel.load_weights(graph_path + "model_checkpoints/LSTM_"+figure_string)
        """
        lstm_end = time.time()
        lstm_dur = lstm_end - lstm_start

        # Predictia pulsului
        # LSTM:
        lstm_pred_start = time.time()
        LSTMdataOutput_predict = LSTMmodel.predict(dataInput_test)

        lstm_gru_intermediary_time = time.time()

        # GRU:
        GRUdataOutput_predict = GRUmodel.predict(dataInput_test)
        gru_pred_end = time.time()

        #Calculul timpilor pentru predictie
        lstm_pred_dur = lstm_gru_intermediary_time - lstm_pred_start
        gru_pred_dur = gru_pred_end - lstm_gru_intermediary_time

        #Evaluarea performantelor pe setul de test
        #LSTMmodel.evaluate(dataInput_test, dataOutput_test, verbose = 2)
        #GRUmodel.evaluate(dataInput_test, dataOutput_test, verbose = 2)


    #Denormalizarea datelor
    LSTMfinalOutput_predict =scaler_hr.inverse_transform(LSTMdataOutput_predict)
    GRUfinalOutput_predict = scaler_hr.inverse_transform(GRUdataOutput_predict)
    dataOutput_test_denorm = scaler_hr.inverse_transform(dataOutput_test)

    #Salvarea modelelor pe hard-disk
    LSTMmodel.save_weights(graph_path + "model_checkpoints/LSTM_"+figure_string)
    GRUmodel.save_weights(graph_path + "model_checkpoints/GRU_"+figure_string)

    #Detectia depasirii pragurilor
    detectie_puls = np.zeros(GRUfinalOutput_predict.shape)
    detectie_puls[GRUfinalOutput_predict > prag_puls_high] = 1 #Momentele in unde se depaseste pulsul maxim dorit
    detectie_puls[GRUfinalOutput_predict < prag_puls_low] = -1 # momentele in care se scade sub pulsul minim dorit

    #Recomandarea modificarii intensitatii antrenamentului ca grafic. antr_rec = primul antrenament unde se depasesc pragurile
    antr_rec = detectie_puls[np.where(GRUfinalOutput_predict > prag_puls_high)[0][0]]
    plt.figure("Rec. puls")
    plt.plot(antr_rec, linewidth=3, linestyle='None', marker='.')
    plt.xlabel("Timp", fontsize=40)
    plt.xticks(fontsize=40)
    plt.title('Recomandari pe baza detectiei momentelor de depasire a intervalului dorit pentru puls', fontsize=40)
    plt.yticks([-1, 0, 1], (
    'Creste\nintensitatea\nantrenamentului', 'Mentine\nintensitatea', 'Scade\nintensitatea\nantrenamentului'),
               rotation=0, fontsize=40)
    plt.show()
    #plt.savefig(graph_path + "GRU_rec_" + figure_string)

################## Statistici si grafice, salvate in locatia graph_path ################################################

    print("Timp de antrenare (LSTM): " + str(lstm_dur) +  " / " + str(len(LSTMmodel.history.history['val_loss'])) + " epoci" +
          "\nTimp de antrenare (GRU):  " + str(gru_dur) + " / " + str(len(GRUmodel.history.history['val_loss'])) + " epoci",file = file_path)
    print("Eroare medie patratica finala pe setul de validare (LSTM): " + str(LSTMmodel.history.history['val_loss'][-1]) +
          "\nEroare medie patratica finala pe setul de validare (GRU):  " + str(GRUmodel.history.history['val_loss'][-1]),
          file = file_path)
    LSTM_vl = np.array(LSTMmodel.history.history['val_loss'])
    GRU_vl = np.array(GRUmodel.history.history['val_loss'])
    print("Eroare patratica minima pe setul de validare (LSTM): " + str(LSTM_vl.min()) + " la epoca " + str(LSTM_vl.argmin()) +
          "\nEroare patratica minima pe setul de validare (GRU):  " + str(GRU_vl.min()) + " la epoca " +  str(GRU_vl.argmin()),
          file = file_path)

    print("Durata medie intre 2 esantioane: " + str(np.average(dif)), file = file_path)
    print("Timpul predictiei / Timp per esantion:",file = file_path)
    print("LSTM: " + str(lstm_pred_dur) + " / " + str(lstm_pred_dur/500) + " per esantion",
          file = file_path)
    print("GRU: " + str(gru_pred_dur) + " / " + str(gru_pred_dur/500) + " per esantion",
          file = file_path)
    file_path.close()

    plt.figure("Precizie LSTM")
    plt.title('Precizie LSTM')
    plt.plot(dataOutput_test_denorm[workout,:])
    plt.plot(LSTMfinalOutput_predict[workout,:])
    plt.legend(('Puls real','Predictie LSTM'))
    plt.xlabel('Timp')
    plt.ylabel('Puls')
    plt.savefig(graph_path + "LSTM_" + figure_string)

    plt.figure("Precizie GRU")
    plt.title('Precizie GRU')
    plt.plot(dataOutput_test_denorm[workout, :])
    plt.plot(GRUfinalOutput_predict[workout, :])
    plt.legend(('Puls real', 'Predictie GRU'))
    plt.xlabel('Timp')
    plt.ylabel('Puls')
    plt.savefig(graph_path + "GRU_" + figure_string)

    plt.figure("Eroare medie patratica LSTM")
    plt.plot(LSTMmodel.history.history['val_loss'])
    plt.plot(LSTMmodel.history.history['loss'])
    plt.title('Eroare medie patratica pentru LSTM')
    plt.legend(('Eroare medie patratica pe setul de validare','Eroare medie patratica pe setul de antrenare'))
    plt.xlabel('Epoca')
    plt.ylabel('Eroare absoluta')
    plt.savefig(graph_path+"LSTM_loss_"+figure_string)

    plt.figure("Eroare medie patratica GRU")
    plt.title('Eroare medie patratica pentru GRU')
    plt.plot(GRUmodel.history.history['val_loss'])
    plt.plot(GRUmodel.history.history['loss'])
    plt.legend(('Eroare medie patratica pe setul de validare', 'Eroare medie patratica pe setul de antrenare'))
    plt.xlabel('Epoca')
    plt.ylabel('Eroare absoluta')
    plt.savefig(graph_path + "GRU_loss_" + figure_string)

    plt.figure('Detectie praguri')

if __name__=="__main__":
    main()