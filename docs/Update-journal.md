2023/10/13: 
    
    Add "README.md" in the court-info-analysis.
    
    Fix the skipping frames bugs.

    Accelerate process by skipping some valid frames in the video.
    
    Finding valid frame using bisection so that the tool can process video faster.    

    Fix the frame count bug. (ori-version start from frame 1). 

    Fix the clipvideo fps which is different from the ori-video bug. 

2023/10/14:

    Fix the utils.write_json bug. 

    Add more information record about court and add new directory in courts.

    Add 3 test videos.

2023/10/15:

    Delete check_top_bot_court hyper-parameters. 

    Fix the bug about bisection algorithm because cv2 will Incorrectly estimated total number of frames. 

    Add more constraint in pre-process function to avoid detect wrong court for one seconds.

2023/10/18:

    Support manual selection of valid frames, you can use run "FrameSelect.py" and save the valid frame in references folder. 

2023/10/19:

    Modified the content of 'res/court/court_kp' form. 

2023/10/22:
 
    Add the "NetDetect.py" for net detection.
    
    Make "FrameSelect.py" select frames faster by selecting from the center. 

    Fix the "CourtDetect.py", "NetDetect.py" and "FrameSelect.py" bugs.

    Delete the court-info-analysis folder.   

2023/10/23:

    Delete the model folder, you can download in https://drive.google.com/drive/folders/16mVjXrul3VaXKfHHYauY0QI-SG-JVLvL?usp=sharing

2023/10/27:

    Adding Terminal Passing Parameters. 

    Modified project file structure. 

    Separate detection from keypoint mapping, split 'main.py' into two files: 'main.py' and 'FrameSelect.py'.

2023/10/28:

    Integrated tracknetv2-pytorch.

2023/10/29:

    Integrated trajectory noise reduction, trajectory display, and hitting frame capture. 

    Modify the type and number of hyperparameters. 

    Add the video information record. 

    Clear the polyfit Rankwarning information. 

    Fix the "VideoDraw.py" the bug of deleting processed videos. 


2023/10/30:

    Add yt-dlp shell script for ShullteSet dataset video downloads. 

2023/10/31：

    Fix "utils.find_references" bugs and make the "FrameSelect.py" easier to use. Now can Adaptive display.

    Fix the list index out of range in denoise.py. 

    Revise the "utils.extract_numbers.py" function.

    Replace the shell program for the dataset download with a python program. 

    Remove the "FrameSelect.py" "yt-dlp4ShuttleSet.py" and "VideoDraw.py" to the "src/tools". 

    Fix the some bugs about between json write and pandas value type int68.  

    Fix the traj2img file name bug. 

2023/11/01:

    Refinement of exception throwing in the ball_detect section. 

2023/11/03:

    Fix the bugs on yt-dlp4ShuttleSet.

2023/11/06:

    Fix the ShuttleSet22 "data_process.py" bug. 

    Transform the original dataset to the match dataset, and update the "yt-dlp4ShuttleSet22.py" program.

    Fix intsall environment.

2023/11/07:

    Correct the player positin. 

    Update "yt-dlp4ShuttleSet.py". 

2023/11/09:

    Update the "FrameSelect.py" and add some trick to correct the net bottom position.

2023/11/10:

    Add the output schema.

    Fix the bug on "event_detect.py".

    Adjust the README.md.

2023/11/11:

    Add trajectory filter and put all reprocess program to "src/reprocess" folder.

    Fix the "smooth.py" and "event_detect.py" bugs.

2023/11/12:

    Add pose interpolation.

    Improve "data4drl,py".

    Fixing the match dataset. 

2023/11/13:

    Upload data4drl dataset, There are 1654 rallies now from 23 matches.

    Adding an Exception Capture Log.

    Fix smooth bugs.

2023/11/18:

    Update the "data4dacreg.py" program. 
    
    Upload data4dacreg dataset, There are 1654 rallies now from 23 matches.

    Revise the data4drl dataset format.

2023/11/20:

    Update the dl hit frame approaches.

    Update data4drl dataset rally number to 2497 and data4acreg 2022.

2023/11/28:

    Update dataset and add new drl environment.
    

    