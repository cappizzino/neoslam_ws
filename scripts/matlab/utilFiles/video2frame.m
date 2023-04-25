%*******************************************************
% COPPE/UFRJ - Federal University of Rio de Janeiro
% Project title: Neocortex SLAM
% Author: Carlos Pizzino
% Description: Save frames from video
% Open Source Code
%*******************************************************
clear vars; clc; close all;

%video_dir = 'C:\dataset\St_Lucia\100909_0845';
video_dir = 'C:\dataset\St_Lucia\180809_1545';
file_type = '.avi';  

videoPath = strcat(video_dir,'\','webcam_video',file_type);
shuttleVideo = VideoReader(videoPath);

ii = 1;
while hasFrame(shuttleVideo)
   img = readFrame(shuttleVideo);
   filename = [sprintf('%04d',ii) '.jpg'];
   fullname = fullfile(video_dir,'frames',filename);
   imwrite(img,fullname)
   ii = ii+1;
end