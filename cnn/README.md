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
