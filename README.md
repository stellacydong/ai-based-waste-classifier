# cv_yolo_scaffold
A scaffold for deploying dockerized flask applications showcasing student computer vision projects using yolo.

### File Structure
The files/directories which you will need to edit are **bolded**, and the files you may need to edit are *italicized*.
DO NOT TOUCH OTHER FILES.

- .gitignore
- Dockerfile
- READMD.md
- app/
     - ai.py
     - **main.py**
     - *requirements.txt*
     - utils.py
     - uwsgi.ini
     - images/
     - static/
          - **images/**
          - *home.css*
          - *Results.css*
          - jquery.js
          - nicepage.css
          - nicepage.js
     - templates/
          - **home.html**
          - *results.html*
     - weights/
          - **https://drive.google.com/file/d/1SeUUbzGjpu74lNm6iYIlggE1tiuHbk3O/view?usp=sharing** 
### ai.py ###
Contains functions used by main.py for working with opencv and running the model on uploaded images.
### main.py ###
Contains the main flask app itself.
### requirements.txt ###
Contains list of packages and modules required to run the flask app. Edit only if you are using additional packages that need to be pip installed in order to run the project.
### static/ ###
Contains the static images, CSS, & JS files used by the flask app for the webpage. Home.css is for the landing page, Results.css is for the landing page. Place all your images used for your website in static/images/.
### templates/ ###
Contains the HTML pages used for the webpage. Edit these to fit your project. home.html is the landing page, results.html is the result page after uploading the image.
### yolo/ ###
Contains info for yolo. Edit yolo.cfg according to the yolov3 darknet instructions. Edit yolo.names according to your labels for your project's objects. **In order to test the flask app locally**, you also need to put your model weights file in this directory and name it as ***yolo.weights***.
### Files used for deployment ###
`Dockerfile`
`uwsgi.ini`
Do not touch these files.
