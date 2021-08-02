# Micro_GRU

#### Contact: Xingjian Chen (xingjchen3-c@my.cityu.edu.hk)

Human Host Status Inference from Temporal Microbiome Changes via Deep Learning


We propose a microbiome host status prediction method using deep learning to infer the host status on the basis of raw time series abundance data with its aggregation on the phylogenetic tree, as well as s series of feature extraction algorithms to reduce the dimension and reconstruct the feature representation of the input variables. 
We evaluate the performance of our model on 6 representative classification tasks from 3 real experimental studies based on 16S rRNA amplicon sequencing. We also compare our method to a series of machine learning-based feature selection and different deep learning architectures. 


How to use our algorithms to bulid the model:


For example:



# 1. Prepare for your config file:


The input of our algorithms are the same with MITRE, so you can see the detailed descriptions from : https://github.com/gerberlab/mitre/blob/master/docs/manual.pdf

also, here we will offer a rough description and the example file (/raw_data).  

Specially, comparing with MITRE, our algorithms only need four main files while MITRE needs 8 files, using david dataset for example:

abundance_data =raw_data/david/abundance.csv

sample_metadata =raw_data/david/sample_metadata.csv

subject_data = raw_data/david/subject_data.csv

jplace_file = raw_data/david/placements.jplace



Input :

1. Abundance table:

The abundance data should be provided as a comma-separated table, with the ﬁrst row providing OTU IDs and the ﬁrst column providing sample IDs.

2. Sample metadata table:

The sample metadata table speciﬁes an associated subject ID and timepoint for each sample ID. It should be given as a comma-separated table with no header row. No particular ordering is expected.

3. Subject data table:

The subject data table gives information about each subject, (including the value of whatever variable will be used as the outcome that MITRE will try to predict, though that does not need to be explicitly marked.) It should be given as a comma-separated table with a header row, whose ﬁrst column is the subject ID. (The ﬁrst ﬁeld in the header row is ignored.) Values may be either strings or numbers (but variables that appear to be Boolean may be converted to 0/1.

4. Pplacer results
This should be the .jplace ﬁle created by placing the representative seuqences from each OTU on a reference tree of known 16S dequences using the pplacer package in maximum likelihood mode.

5. config file, the config file is to load the data and set the special data preprocessing and comparision parameters:
    
    [benchmarking]
    
    [description]
    tag = david   # name of dataset
    
    [data]
    outcome_variable = diet    # name of label 
    outcome_positive_value = Plant  
    taxonomy_source = hybrid
    
    abundance_data =raw_data/david/abundance.csv
    sample_metadata =raw_data/david/sample_metadata.csv
    subject_data = raw_data/david/subject_data.csv
    jplace_file = raw_data/david/placements.jplace
    
    [preprocessing]   # filtering steps
    time_point=8  
    min_overall_abundance = 10  
    min_sample_reads = 5000
    take_relative_abundance = True
    aggregate_on_phylogeny = True
    log_transform = False
    
    temporal_abundance_threshold = 0.001
    temporal_abundance_consecutive_samples = 2
    temporal_abundance_n_subjects = 4
    discard_surplus_internal_nodes = True
    
    [comparison_methods]
    n_intervals = 10
    n_consecutive = 3




# 2. Gnenrate your datasets and save as the npy file:
 After getting ready for the input data, using
```sh
python3 data_generate.py -fn david
```
to gnenrate your datasets and save as the npy file. 'david' is the file name (name of the datasets).

The output should be something like:

Time Windows:  [(0.0, 400.0), (200.0, 600.0), (400.0, 800.0), (600.0, 1000.0)]
Time start:  0.0
Time end:  1000.0
Comb:  (17, 56, 4)
Label length:  17

Where 'Comb' means a 3D matrix, the first number is the size of the samples, and the second is the number of OTU features. The third is the number of time points.

# 3. Evaluation

After the step 2, you will get 2 npy files for your dataset. One is the feature file and the other is the label file.

if you want to test the performance based on different deep learning models:
You can run:

```sh
python3 com_dl.py -fn david -clf mlp -rs 1 -es tr
```
'clf' means the chosen deep learning architectures which have been described in our manuscript, we offer mlp, gru1, gru2,gru3,gru4,gru5, and gru6 for evaluation.

'rs' is the repeat time for each classifier since some classifiers will obtain different result each time, therefore you can run several times and get average prediction results.

'es' means whether you want to use the earlystopping on training data (tr) or validation data (te).


if you want to test the performance of tranditional classifiers:
You can run:

```sh
python3 com_tc.py -fn david -clf l1 -rs 1
```
'clf' means the choosen classifiers, we offer rf, l1, svm, knn and xgb for evaluation.


if you want to test the performance based on different feature selection methods:
You can run:

```sh
python3 com_fs.py -fn david -clf l1 -rs 1 -es tr
```
'clf' means the choosen methods for feature extraction, we offer ae, vt, pca, rfe, l1 and rf for evaluation.
The basic deep learning model is GRU4.


if you want to test the performance of transer learning:
You can run:

```sh
python3 fine_tuning.py -fn david -clf tcrf -rs 1 -es tr
```
'clf' means the choosen fine-tuning models, we offer tcrf, tcsvm, ft1,ft2 and ft3 for comparison.

All the model parameters and the structures can be seen from our paper.





