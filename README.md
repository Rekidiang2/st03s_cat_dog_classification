# Diabates Prediction (WebApp)

<img src="/images/capture_app.jpg" width="1000" height="300" />


In healthcare field diagnose a problem early offer more chance for traitement and guerison in this project we apply machine learning techniques to predict whether a patient will develop diabetes within the next five years. Early detection and diagnosis of diabetes is that the early stages of diabetes are often non-symptomatic. People who are on the path to diabetes (also known as prediabetes) often do not know that they have diabetes until it is too late.

## Contents

1. Project Structure
2. Prosess
3. How to run
4. Link
5. To improve
6. About Me

## 1. Project Structure

#### Data
* ├── diabetes.csv
* ├── cleaned_data.csv
#### images
* ├── contains images, graph and figures
#### models
* ├── contains trained and used models for prediction

##### annex.py
##### app.py
##### diabetes_result.db (contains predictions results)
##### Procfile and setup.sh files (dependencies for deployment on heroku)
##### environment
##### requirements.txt
##### gitignore

## 2. Process

* step1 :  build application structure with streamlit
* step2 :  Load pre-trained model
* step3 :  insert features
* step4 :  make prediction and save result in the database

## 3. How to run

### 3.1. CLONE PROJECT DIRECTORY

+ $ git clonehttps://github.com/RekidiangData-S/wap01s_diabetes_prediction.git
+ $ cd wap01s_diabetes_prediction

### 3.2. CREATE & ACTIVATE VIRTUAL ENVIRONMENT

#### 3.2.1. WITH CONDA

+ Verify if you have conda installed ($conda --version) if not go to [anconda](https://www.anaconda.com/products/individual) or [miniconda](https://docs.conda.io/en/latest/miniconda.html) to download and install it

+ $ conda env create -f environment.yml
+ $ conda activate wap01s_venv <= Activate virtual Environment
+ + $ conda deactivate  <= Deactivate virtual Environment

#### 3.2.2. WITH PIP

##### (Windows) 
+ $ python -m venv wap01s_venv 
+ $ wap01s_venv\Scripts\activate <= Activate virtual Environment
+ $ deactivate <= Deactivate virtual Environment
+ $ pip install -r requirements.txt
+ $ streamlit run app.py
##### (MasOS || LINUX)
+ $ python3 -m venv wap01s_venv 
+ $ source wap01s_venv/bin/activate <= Activate virtual Environment  
+ $ deactivate <= Deactivate virtual Environment
+ $ pip install -r requirements.txt
+ $ streamlit run app.py

## 4. Link

+ [WebApp with Streamlit]()

## 5. To improve

+ deploy with heroku
+ insert logo (voir deleted fold)
+ set caution message
+ push on github
+ finalize analysis page
+ finalize about page
+ readme image 
+ social media preview (Setting -> preview)

## 6. About Me
___

### I'm a data and technology passionate person, Artificial Intelligence enthusiast 

> My Website [Click Here](https://kiesediangebeni/github.io)

> Social Network

[![alt text][1.1]][1]
[![alt text][2.1]][2]
[![alt text][3.1]][3]
[![alt text][4.1]][4]

[1.1]: https://i.imgur.com/oFsAcMx.png (facebook icon with padding)
[2.1]: https://i.imgur.com/YCdR3o9.png (twitter icon with padding)
[3.1]: https://i.imgur.com/5BWvIrF.png (github icon with padding)
[4.1]: https://i.imgur.com/UA7Oh6z.png (medium icon with padding)

[1]: http://www.facebook.com/reagan.kiese.37
[2]: https://twitter.com/ReaganKiese
[3]: https://github.com/RekidiangData-S
[4]: https://medium.com/@rkddatas

