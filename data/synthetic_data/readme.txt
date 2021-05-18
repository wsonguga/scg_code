In the data file, each row includes 1000 sensor readings (100Hz * 10 seconds) + heart_rate + systolic + diastolic
    synthetic_train_set.npy
    synthetic_test_set.npy

Write a stream data analytics program (signal processing + machine learning) to build the relationship model between the sensor data and the three parameters (Systolic, Diastolic, Heart_rate), 
and predict the future four parameters per second from future raw sensor data. Python language 
is expected for the stream data analytics program. The goal is to achieve MAE <=3 for all three parameters (S, D, H).
