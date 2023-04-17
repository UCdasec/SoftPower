# Training
To train the model use the following


```python
python3 cnn/train.py -i Input Data Set  -o Model Output Directory -aw Attack Window (startIndex_endIndex) -lm HW/ID 
-tb (target byte) -e (Number of Epochs) -tn (Number of Train Traces)
```

# Testing 
To test a given trained model


```python
python3 cnn/test.py -i Input Data Set  -m Model Output Directory -o Output Directory for Results 
-aw Attack Window (startIndex_endIndex) -lm HW/ID -tb (target byte) -tn (Number of Test Traces)
```
# Hyperparameters of CNN
<img width="323" alt="Screenshot 2023-04-17 at 9 54 37 AM" src="https://user-images.githubusercontent.com/54579704/232505355-cd7b21c0-5013-45a2-abff-921d5f2f1256.png">



# The system model of machine-learning profiling side-channel attacks with software discrepancies
<img width="369" alt="Screenshot 2023-04-17 at 9 57 12 AM" src="https://user-images.githubusercontent.com/54579704/232506078-71cb7192-c159-41e9-afcf-d442e1e39564.png">
