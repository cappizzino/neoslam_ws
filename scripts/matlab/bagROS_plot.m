close all
clearvars

exp = 'outdoor_afternoon'; % 'robotarium'; % 'corridor' %'outdoor_afternoon'

filesSaved = 1;
plotConfig = 1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
switch exp
    case 'robotarium'
        file =  'E:\dataset\output\robotarium\husky_out_cnn.bag';
        offset1 = 10;
        offset2 = 20;
        theta = -58;%-56;
    case 'corridor'
        file = 'E:\dataset\output\corridor\husky_out.bag';
        offset1 = 1;
        offset2 = 1;
        theta = 0;
    case 'outdoor_afternoon'
        file = 'E:\dataset\output\outdoor_afternoon\husky_out_after01.bag';
        offset1 = 1;
        offset2 = 1;
        theta = 60;
    otherwise
        warning('No plot created.')
        return
end

bag = rosbag(file);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
data_odom = select(bag,'Topic','/husky_hwu/odom');
pose_odom = timeseries(data_odom, 'Pose.Pose.Position.X',...
    'Pose.Pose.Position.Y', 'Pose.Pose.Position.Z');

%frames = bag.AvailableFrames;
%tf = getTransform(bag,'odom','base_link');
%q = quaternion(tf.Transform.Rotation.X,tf.Transform.Rotation.Y,...
%    tf.Transform.Rotation.Z,tf.Transform.Rotation.W);
%%eul = eulerd(q,'XYZ','frame')
%rotm_odom = quat2rotm(q);
%tra = [tf.Transform.Translation.X tf.Transform.Translation.Y];

pose_odom = [pose_odom.Data(:,1), pose_odom.Data(:,2), ...
    pose_odom.Data(:,3)];

R = [cosd(theta) -sind(theta); sind(theta) cosd(theta)];
pose_odom_frame = R*pose_odom(offset1:end-offset2,1:2)';
pose_odom_frame = [pose_odom_frame-pose_odom_frame(:,1)]';

f1 = figure(1);
%subplot(1,2,1);
plot(pose_odom_frame(:,1),pose_odom_frame(:,2));
title('Husky Position Odometry')
xlabel('x(m)')
ylabel('y(m)')
grid on
grid minor

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if plotConfig == 1
    width = 3;      % Width in inches
    height = 3;     % Height in inches
    alw = 0.75;     % AxesLineWidth
    fsz = 9;        % Fontsize
    lw = 1.5;       % LineWidth
    msz = 8;        % MarkerSize

    pos = get(f1, 'Position');
    ff1 = get(f1,'Children');

    set(f1, 'Position', [pos(1) pos(2) width*100, height*100]);
    set(ff1, 'FontSize', fsz, 'LineWidth', alw); %<- Set properties

    fileName = fullfile(pwd,'outputFiles', exp);
    figname11 = [strcat(fileName,'\',exp,'_pos') '.jpg'];
    figname12 = [strcat(fileName,'\',exp,'_pos_eps') '.eps'];

    if filesSaved == 1
        exportgraphics(f1,figname11)
        exportgraphics(f1,figname12)
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
bSe1_htm = select(bag,'Topic','/feats_htm');
msg_htm = readMessages(bSe1_htm,'DataFormat','struct');
D1 = cellfun(@(m) find(m.Data),msg_htm,'UniformOutput',false);
S = evaluateSim(D1(offset1:size(msg_htm,1)-offset2),D1(offset1:size(msg_htm,1)-offset2),'wincell');

figure(2)
s = imagesc(S);
colorbar
title('Confusion Matrix')
xlabel('Query images')
ylabel('Database images')

size_gt = size(S);
GT = eye(size_gt(1,1));
tole = 10;
rep = 3;
vpr = 120;

for j = 1:4
    for i=1:size_gt(1,1)
        pos_gt = i + (j-1)*vpr;
        if (pos_gt+tole)>size_gt(1,1)
            break
        end
        GT(i,pos_gt:(pos_gt+tole))=1;
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if plotConfig == 1
    width = 4;      % Width in inches
    height = 3;     % Height in inches
    alw = 0.75;     % AxesLineWidth
    fsz = 9;        % Fontsize
    lw = 1.5;       % LineWidth
    msz = 8;        % MarkerSize

    f2 = figure(2);
    pos = get(f2, 'Position');
    ff2 = get(f2,'Children');

    set(f2, 'Position', [pos(1) pos(2) width*100, height*100]);
    set(ff2, 'FontSize', fsz, 'LineWidth', alw); %<- Set properties

    fileName = fullfile(pwd,'outputFiles', exp);
    figname21 = [strcat(fileName,'\',exp,'_conf') '.jpg'];
    figname22 = [strcat(fileName,'\',exp,'_conf_eps') '.eps'];

    if filesSaved == 1
        exportgraphics(f2,figname21)
        exportgraphics(f2,figname22)
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
bSel_vt = select(bag,'Topic','/husky_hwu/LocalView/Template');
msg_vt = readMessages(bSel_vt,'DataFormat','struct');
id = cellfun(@(m) double(m.CurrentId),msg_vt);

figure(3)
plot(id, 'bx')
grid on;
grid minor
%stem(id)
title('View Cells created')
xlabel('Query images')
ylabel('Id View Cells')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if plotConfig == 1
    width = 3;      % Width in inches
    height = 3;     % Height in inches
    alw = 0.75;     % AxesLineWidth
    fsz = 9;        % Fontsize
    lw = 1.5;       % LineWidth
    msz = 8;        % MarkerSize

    f3 = figure(3);
    pos = get(f3, 'Position');
    ff3 = get(f3,'Children');

    set(f3, 'Position', [pos(1) pos(2) width*100, height*100]);
    set(ff3, 'FontSize', fsz, 'LineWidth', alw); %<- Set properties

    fileName = fullfile(pwd,'outputFiles', exp);
    figname31 = [strcat(fileName,'\',exp,'_vc') '.jpg'];
    figname32 = [strcat(fileName,'\',exp,'_vc_eps') '.eps'];

    if filesSaved == 1
        exportgraphics(f3,figname31)
        exportgraphics(f3,figname32)
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% GT = GT|GT';
% [P, R, F1] = createPR(S, GT);
% plot(R,P);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
data_em = select(bag,'Topic','/husky_hwu/ExperienceMap/Map');
msg = readMessages(data_em,'DataFormat','struct');
em = msg{data_em.AvailableTopics.NumMessages};
 
x = em.Node(:);
n = em.NodeCount;
y0 = double(vertcat(x.Id));
y1 = vertcat(x.Pose)';
z1 = vertcat(y1.Position);
z2 = vertcat(y1.Orientation);
w = vertcat([z1.X; z1.Y; z1.Z; z2.X; z2.Y; z2.Z; z2.W])';
pose = [y0 w];

x = em.Edge(:);
nl = em.EdgeCount;
y0 = vertcat(x.Id);
y1 = vertcat(x.SourceId);
y2 = vertcat(x.DestinationId);
links = [y0 y1 y2];

markersize = 5;
linewidth = 2;
draw_links = 1;

%pose_em(:,2) = pose(offset1:end-offset2, 2)-pose(1,2);
%pose_em(:,3) = pose(offset1:end-offset2, 3)-pose(1,3);

pose_em(:,2) = pose(offset1:end-offset2, 2);
pose_em(:,3) = pose(offset1:end-offset2, 3);

f4 = figure(4);
%subplot(1,2,2);
plot(pose_em(:,2), pose_em(:,3), 'go', 'MarkerSize', markersize, 'MarkerFaceColor', 'none', 'LineWidth', linewidth)
%hold on
%plot(pose_odom_frame(:,1)-pose_odom_frame(1,1),pose_odom_frame(:,2)-pose_odom_frame(1,2));
%axis equal;
title('Husky Position Map Experience')

if draw_links == 1
	if nl > 0
        for k = offset1:nl
            sn = links(k, 2) + 1;
            dn = links(k, 3) + 1;
            line([pose(sn, 2) pose(dn, 2)], [pose(sn, 3) pose(dn, 3)]);                
        end
	end
end

xlabel('x(m)');
ylabel('y(m)');
grid on;
grid minor

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if plotConfig == 1
    width = 3;      % Width in inches
    height = 3;     % Height in inches
    alw = 0.75;     % AxesLineWidth
    fsz = 9;        % Fontsize
    lw = 1.5;       % LineWidth
    msz = 8;        % MarkerSize

    pos = get(f4, 'Position');
    ff4 = get(f4,'Children');

    set(f4, 'Position', [pos(1) pos(2) width*100, height*100]);
    set(ff4, 'FontSize', fsz, 'LineWidth', alw); %<- Set properties

    fileName = fullfile(pwd,'outputFiles', exp);
    figname41 = [strcat(fileName,'\',exp,'_em') '.jpg'];
    figname42 = [strcat(fileName,'\',exp,'_em_eps') '.eps'];

    if filesSaved == 1
        exportgraphics(f4,figname41)
        exportgraphics(f4,figname42)
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% width = 3;      % Width in inches
% height = 3;     % Height in inches
% alw = 0.75;     % AxesLineWidth
% fsz = 9;        % Fontsize
% lw = 1.5;       % LineWidth
% msz = 8;        % MarkerSize
% 
% pos = get(f1, 'Position');
% ff1 = get(f1,'Children');
% 
% set(f1, 'Position', [pos(1) pos(2) width*100, height*100]);
% set(ff1, 'FontSize', fsz, 'LineWidth', alw); %<- Set properties
% 
% fileName = fullfile(pwd,'outputFiles', exp);
% figname11 = [strcat(fileName,'\',exp,'_em') '.jpg'];
% figname12 = [strcat(fileName,'\',exp,'_em_eps') '.eps'];
% 
% if filesSaved == 1
% 	exportgraphics(f1,figname11)
% 	exportgraphics(f1,figname12)
% end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
data_em_pose = select(bag,'Topic','/husky_hwu/ExperienceMap/RobotPose');
msg_em = readMessages(data_em_pose,'DataFormat','struct');
x_em_pose = cellfun(@(m) double(m.Pose.Position.X),msg_em);
y_em_pose = cellfun(@(m) double(m.Pose.Position.Y),msg_em);

f5 = figure(5);
plot(x_em_pose(:,1),y_em_pose(:,1));
hold on
plot(pose_odom_frame(:,1),pose_odom_frame(:,2));
title('Husky Position Odometry')
xlabel('x(m)')
ylabel('y(m)')
grid on
grid minor

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
data_info = select(bag,'Topic','/info');
info = readMessages(data_info,'DataFormat','struct');
info_id = cellfun(@(m) uint32(m.CurrentViewCell),info,'UniformOutput',false);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% bSe_im = select(bag,'Topic','/husky_hwu/processed_image');
% msg_im = readMessages(bSe_im);
% 
% % for j=1:size(msg_im,1)
% % 
% %     if j > offset1
% %         [img_d,alpha_d] = readImage(msg_im{j});
% %         [img_q,alpha_q] = readImage(msg_im{id(j)});
% %     
% %         position =  [1 50];
% %     
% %         husky_image_d = insertText(img_d,position,j,'AnchorPoint','LeftBottom');
% %         husky_image_q = insertText(img_q,position,id(j),'AnchorPoint','LeftBottom');
% %     
% %         figure(4)
% %         imshow([husky_image_d,husky_image_q])
% %         title('Images');
% %     
% %         pause(1)
% %         clear husky_image_d husky_image_q
% %     end
% % end
% 
% %(A) Experiment environment. (B) Visual odomerty. 
% %(C) Experience map based on RatSLAM. (D) Experience map of FT model-based RatSLAM.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% bSel_ta = select(bag,'Topic','/husky_hwu/PoseCell/TopologicalAction');
% msg_ta = readMessages(bSel_ta,'DataFormat','struct');
% 
% actions = cellfun(@(m) [m.Action, m.SrcId, m.DestId], msg_ta, 'UniformOutput', false);
% actions = cell2mat(actions);
% actions_vp_idx = uint16(find(actions(:,1)==2));
% actions_vp = actions(actions_vp_idx,:);
% 
% vpr_merge = select_vp(actions_vp);
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% i=24; j=147;
% 
% f4 = figure(4);
% imshow(select_img(i, j, msg_im))
% title('True Positive Image');
% 
% width = 7;      % Width in inches
% height = 3;     % Height in inches
% alw = 0.75;     % AxesLineWidth
% fsz = 9;        % Fontsize
% lw = 1.5;       % LineWidth
% msz = 8;        % MarkerSize
% 
% pos = get(f4, 'Position');
% ff4 = get(f4,'Children');
% 
% set(f4, 'Position', [pos(1) pos(2) width*100, height*100]);
% set(ff4, 'FontSize', fsz, 'LineWidth', alw); %<- Set properties
% 
% fileName = fullfile(pwd,'outputFiles', exp);
% figname41 = [strcat(fileName,'\',exp,'_tp') '.jpg'];
% figname42 = [strcat(fileName,'\',exp,'_tp_eps') '.eps'];
% 
% if filesSaved == 1
% 	exportgraphics(f4,figname41)
% 	exportgraphics(f4,figname42)
% end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% i=54; j=163;
% figure(5)
% imshow(select_img(i, j, msg_im))
% title('True Positive Image');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% i=64; j=170;
% 
% f6 = figure(6);
% imshow(select_img(i, j, msg_im))
% title('False Positive Image');
% 
% width = 7;      % Width in inches
% height = 3;     % Height in inches
% alw = 0.75;     % AxesLineWidth
% fsz = 9;        % Fontsize
% lw = 1.5;       % LineWidth
% msz = 8;        % MarkerSize
% 
% 
% pos = get(f6, 'Position');
% ff6 = get(f6,'Children');
% 
% set(f6, 'Position', [pos(1) pos(2) width*100, height*100]);
% set(ff6, 'FontSize', fsz, 'LineWidth', alw); %<- Set properties
% 
% fileName = fullfile(pwd,'outputFiles', exp);
% figname61 = [strcat(fileName,'\',exp,'_fp') '.jpg'];
% figname62 = [strcat(fileName,'\',exp,'_fp_eps') '.eps'];
% 
% if filesSaved == 1
% 	exportgraphics(f6,figname61)
% 	exportgraphics(f6,figname62)
% end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% places = [actions_vp(:,2); actions_vp(:,3)];
% places = sort(places);
% places = unique(places);
% places_number = size(places,1);
% place_cell = {};
% 
% for i=1:places_number
%     place_cell(i,1) = {places(i)};
%     
%     places_cell_idx = find(actions_vp(:,2)==places(i));
%     place_cell{i,2} = actions_vp(places_cell_idx,3)';
%     
%     places_cell_idx = find(actions_vp(:,3)==places(i));
%     place_cell{i,3} = actions_vp(places_cell_idx,2)';
% end
% 
% vpr = {};
% 
% for i=1:places_number 
%     c = [place_cell{i,1},place_cell{i,2},place_cell{i,3}];
%     c = sort(c);
%     c = unique(c);
%     vpr{i} = c;
% end
% 
% inter = eye(places_number);
% 
% for i=1:places_number
%     for j=i:places_number
%         aux = intersect(vpr{i},vpr{j});
%         if  ~isempty(aux)
%             %aux2 = [vpr{i},vpr{j}];
%             %aux2 = sort(aux2);
%             %aux2 = unique(aux2);
%             %vpr_merge{w} = aux2;
%             %w = w + 1;
%             inter(i,j) = 1;
%         end
%     end
% end
% 
% vpr_merge = {};
% for i=1:places_number
%     vpr_merge {i} = find(inter(i,:));
% end
