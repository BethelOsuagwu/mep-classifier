# Data
## Training and validation data
1. **training-25-05-2023.csv** - This is a cleaner version of `training_data_with_artefact_class_all_response_win.csv`. It has no isolated background classes between stimulation artefact and response classes. The classes apart from the stimulation artefact class, are more balanced. Also the class labels are improved. 
2. **training_data_with_artefact_class_all_response_win.csv** - This is similar to `training-25-05-2023.csv`. In fact they are from the same dataset. But this one has isolated background classes between stimulation artefact and response classes. It also contain a few bad quality responses. It has significant class imbalance.

## Testing data
2. **testing-18-05-2023.csv** - testing data
The classes are more balanced. As this dataset contains significant stimulation artefact class, this data could perhaps be transferred to training.

## Todo
### [DONE]Better precision labelling
**In EPR recorder**
1. Create a btn that opens a window
2. The window allowes you to scroll through the trials
3. for each trial, you can use 4 checkboxes to state which of the following components are present: stim-artifact|background-pre|EP|background-post.
4. Based on the number of components present, N resizeable color coded rectangles appear  over the signal,
5. resize the rectangles as appriopriate. When a rectangle is resized, the size of adjacent rectagles are up adjusted to ensure that there is no space bween the neighbours.
6. The rectangles are saved in EPR.epochs.training...
7. An export button is used to export the data