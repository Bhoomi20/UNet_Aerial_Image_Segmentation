# UNet_Aerial_Image_Segmentation
Semantic Image Segmentation of Satellite Images of Buildings for Disaster Management

The model has been trained and the weights are stored in "model-dsbowl2018-1.h5" file
Same is used in the Main_Test_Code.py

The code is integrated with the UI in UiServer.py file

# Execution
Open python executable terminal and then execute the below command.

-- pip install -r library-requirement.txt

-- py UiServer.py

this will open a browser in localhost - http://127.0.0.1:5000/
   
# Result
Images before and after disaster can be added using the option given in the UI and click on "Process" button.
On clicking "Process" it will take you to the page with following results:

  Input images  - images provided by the user which is before and after the disaster. Some sample images are present in the images folder for use
    
  Enhanced images - these are the images shown after applying the Contrast limited adaptive histogram equalization(CLAHE)
  
  Segmented images - images after applying the U-net model on the enhanced images
  
  Difference between the segemnted images - this shows the areas of damage by comparing the 2 segmented images and the % of the disaster affected region is given as well.
  
These image results are stored under static/results/<epoch_time_of_upload> for future reference  
