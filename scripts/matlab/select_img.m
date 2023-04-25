function [C] = select_img(i, j, msg_im)
position =  [1 50];

[img_d,alpha_d] = readImage(msg_im{i});
[img_q,alpha_q] = readImage(msg_im{j});

husky_image_d = insertText(img_d,position,i,'AnchorPoint','LeftBottom');
husky_image_q = insertText(img_q,position,j,'AnchorPoint','LeftBottom');

C = [husky_image_d,husky_image_q];