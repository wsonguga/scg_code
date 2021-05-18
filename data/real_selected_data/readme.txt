Each row includes good or bad data (100Hz * 10 seconds) + systolic + diastolic + label (0 - good, 1 - bad)
    classifier_train_data.npy # include both good and bad data

Each row includes data (100Hz * 10 seconds) + systolic + diastolic
    real_train_data.npy  # be considered as good data by human eyes
    real_test_data.npy  # be considered as good data by human eyes


Note: bad data examples by human eyes are not necessarily bad when appying prediction. 
