# Introduction
MATLAB deployment of the meP classifier.
## Installation
Clone the directory and *m/* to matlab path.

Run the following in matlab prompt to ensure the test passes.
```MATLAB
mepclassifier.ClassifierManager().classifier().sanityTest()
```
## Usage
```MATLAB
% Use the default classifier to classify MEP data.
classifier=mepclassifier.ClassifierManager().classifier('default'); 
data=...;
[start, stop, preds]=classifier.classify(data);
```

## Adding custom classifiers
Implement the *mepclassifier.ClassifierContract* contract:
```MATLAB
classdef CustomClassifier < mepclassifier.ClassifierContract
% ...
end
See an example in *m/DefaultMEPClassifier.m*.

Insert an entry in *m/mep_classifier.json* for the implementation with a unique driver name:
```json
{
  "driver":"custom",
  "name":"Custom classifier",
  "classname":"CustomClassifier"
}
```
Use the custom classifier:
```MATLAB
customClassifier=mepclassifier.ClassifierManager().classifier('custom'); 
data=...;
[start, stop, preds]=customClassifier.classify(data);

```
