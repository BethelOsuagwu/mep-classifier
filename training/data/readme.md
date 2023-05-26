# Data
## Class labels
Unless otherwise stated, the class labels are as follows:
- 1 - background
- 2 - stimulation artefact
- 3 - response
## Training and validation data
1. **training-25-05-2023.csv** - This is a cleaner version of `training_data_with_artefact_class_all_response_win.csv`. It has no isolated background classes between stimulation artefact and response classes. The classes apart from the stimulation artefact class, are more balanced. Also the class labels are improved. 
2. **training_data_with_artefact_class_all_response_win.csv** - This is similar to `training-25-05-2023.csv`. In fact they are from the same dataset. But this one has isolated background classes between stimulation artefact and response classes. It also contain a few bad quality responses. It has significant class imbalance.

## Testing data
2. **testing-18-05-2023.csv** - testing data
The classes are more balanced. As this dataset contains significant stimulation artefact class, this data could perhaps be transferred to training.