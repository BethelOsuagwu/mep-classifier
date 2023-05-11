# Introduction
MATLAB deployment of the MEP classifier.
## Installation
Clone the directory and add *m/* to matlab path.

Run the following in matlab prompt and ensure the test passes.
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
```
See an example in *m/DefaultClassifier.m*.

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
