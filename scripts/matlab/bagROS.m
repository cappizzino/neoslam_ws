clearvars
close all

bag = rosbag('C:\dataset\_2022-04-07-14-14-35_robotarium.bag');
theta = -56;%-56; %-146,488119
R = [cosd(theta) -sind(theta); sind(theta) cosd(theta)];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% bSe_tf = select(bag,'Topic','/tf');
% msgStructs_tf = readMessages(bSe_tf,'DataFormat','struct');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
bSe1 = select(bag,'Topic','/feats_htm');
msgStructs = readMessages(bSe1,'DataFormat','struct');
D1 = cellfun(@(m) find(m.Data),msgStructs,'UniformOutput',false);
S = evaluateSim(D1(10:414),D1(10:414),'wincell');

figure(1)
s = imagesc(S);
colorbar
title('Confusion Matrix')
xlabel('Query images')
ylabel('Database images')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
bSe2 = select(bag,'Topic','/husky_hwu/ExperienceMap/MapMarker');
msgStructs2 = readMessages(bSe2,'DataFormat','struct');
a = struct2cell(msgStructs2{bSe2.NumMessages,1});
b = a(28);
d = struct2cell(b{1,1});

x_em = cellfun(@(m) double(m),d(2,:));
y_em = cellfun(@(m) double(m),d(3,:));

% figure(2)
% title('Map')
% plot(pose_em(1,:), pose_em(2,:))

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
bSe3 = select(bag,'Topic','/odometry/filtered');
msgStructs3 = readMessages(bSe3,'DataFormat','struct');

xPoints = cellfun(@(m) double(m.Pose.Pose.Position.X),msgStructs3);
yPoints = cellfun(@(m) double(m.Pose.Pose.Position.Y),msgStructs3);
zPoints = cellfun(@(m) double(m.Pose.Pose.Position.Z),msgStructs3);
pose_odom = [(xPoints)'; (yPoints)'];

A = cellfun(@(m) double(m.Pose.Pose.Orientation.X),msgStructs3);
B = cellfun(@(m) double(m.Pose.Pose.Orientation.Y),msgStructs3);
C = cellfun(@(m) double(m.Pose.Pose.Orientation.Z),msgStructs3);
D = cellfun(@(m) double(m.Pose.Pose.Orientation.W),msgStructs3);
yawPoints = quaternion(A,B,C,D);

% Transformation
t_orientation_odom = yawPoints(1);
rotm_odom = quat2rotm(t_orientation_odom);
tform = rotm2tform(rotm_odom);

t_trans_odom = [xPoints(1), yPoints(1), zPoints(1)];

%pose_odom = R*pose_odom;
%pose_odom = pose_odom;
%pose_odom = pose_odom - pose_odom(:,1);

%pose_em = [(x_em-x_em(1)); (y_em-y_em(1))];
pose_em = [(x_em); (y_em)];
%pose_em = pose_em(:,20:end);
%pose_em = pose_em - pose_em(:,1);

figure(3)
plot(pose_em(1,:), pose_em(2,:), pose_odom(1,:), pose_odom(2,:))
title('Husky Position')
xlabel('x(m)')
ylabel('y(m)')
grid on
grid minor

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
bSe4 = select(bag,'Topic','/husky_hwu/LocalView/Template');
msgStructs4 = readMessages(bSe4,'DataFormat','struct');
id = cellfun(@(m) double(m.CurrentId),msgStructs4);

figure(4)
stem(id)
% for i=1:length(id)
%     plot([i,id(i)],[2,1],'b--o')
%     hold on
% end
title('View Cells created')
xlabel('Query images')
ylabel('Id View Cells')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
plot_pos_01 = select(bag,'Topic','/husky_hwu/ExperienceMap/RobotPose');
plot_pos_02 = select(bag,'Topic','/odometry/filtered');

pose_em_p = timeseries(plot_pos_01, 'Pose.Position.X', 'Pose.Position.Y');
pose_od_p = timeseries(plot_pos_02, 'Pose.Pose.Position.X', 'Pose.Pose.Position.Y');

figure(5)
%plot(pose_em(1,:), pose_em(2,:), x_em_p, y_em_p)
plot(pose_em_p.Data(:,1), pose_em_p.Data(:,2),pose_od_p.Data(:,1), pose_od_p.Data(:,2));
title('Husky Position Exp')
xlabel('x(m)')
ylabel('y(m)')
grid on
grid minor

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%bSe6 = select(bag,'Topic','/tf');
%msgStructs6 = readMessages(bSe6,'DataFormat','struct');

%child = cellfun(@(m) m.Transforms.ChildFrameId,msgStructs6,'UniformOutput',false);
%tf = cellfun(@(m) m.Transforms.Transform,msgStructs6,'UniformOutput',false);
%child_s = cell2struct(child);

frames = bag.AvailableFrames;
tf = getTransform(bag,'odom','base_link');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% bSe5 = select(bag,'Topic','/husky_hwu/processed_image');
% msgStructs5 = readMessages(bSe5);
% 
% for j=1:size(msgStructs5,1)
% 	[img,alpha] = readImage(msgStructs5{j});
%     
%     position =  [1 50; 100 50];
%     value = [j id(j)];
%     kusky_image = insertText(img,position,value,'AnchorPoint','LeftBottom');
%     
%     figure(5)
%     imshow(kusky_image)
%     title('Images');
%     
%     pause(1)
%     clear kusky_image
% end

% xPoints = cellfun(@(m) double(m.Pose.Pose.Position.X),msgStructs);
% yPoints = cellfun(@(m) double(m.Pose.Pose.Position.Y),msgStructs);
% plot(xPoints,yPoints)
% axis([-6 1 -11 1])
% title('Husky Position')
% xlabel('x')
% ylabel('y')
% grid on
% grid minor
% legend('amcl pos', 'Location','southwest')
% 
% bSel1 = select(bag,'Topic','/feats_cnn');
% msgStructs1 = readMessages(bSel1,'DataFormat','struct');
% 
% for i=1:length(msgStructs1)
%     cnn_features(i,:) = msgStructs1{i}.Data;
% end