close all
clearvars
warning('off', 'MATLAB:MKDIR:DirectoryExists');

exp = 'barnbot';
filesSaved = 0;
plotConfig = 0;

file = fullfile('experiments',exp,'exp1.bag');
bag = rosbag(file);

mkdir(fullfile(pwd,'experiments',exp,'outputFiles'));

data_path = select(bag,'Topic','/lio_sam/mapping/path');
path = readMessages(data_path(end),'DataFormat','struct');

data_marker = select(bag,'Topic','/lio_sam/mapping/loop_closure_constraints');
marker = readMessages(data_marker(end),'DataFormat','struct');

% data_odom = select(bag,'Topic','/odom');
% pose_odom = readMessages(data_odom,'DataFormat','struct');

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% data_odom = select(bag,'Topic','/odom');
% pose_odom = timeseries(data_odom, 'Pose.Pose.Position.X',...
%     'Pose.Pose.Position.Y', 'Pose.Pose.Position.Z');
% 
% pose_odom = [pose_odom.Data(:,1), pose_odom.Data(:,2), ...
%     pose_odom.Data(:,3)];
% 
% f1 = figure(1);
% plot(pose_odom_frame(:,1),pose_odom_frame(:,2));
% title('Odometry')
% xlabel('x(m)')
% ylabel('y(m)')
% grid on
% grid minor
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% if plotConfig == 1
%     width = 3;      % Width in inches
%     height = 3;     % Height in inches
%     alw = 0.75;     % AxesLineWidth
%     fsz = 9;        % Fontsize
%     lw = 1.5;       % LineWidth
%     msz = 8;        % MarkerSize
% 
%     pos = get(f1, 'Position');
%     ff1 = get(f1,'Children');
% 
%     set(f1, 'Position', [pos(1) pos(2) width*100, height*100]);
%     set(ff1, 'FontSize', fsz, 'LineWidth', alw); %<- Set properties
% 
%     
%     figname11 = fullfile(pwd,'experiments',exp,'outputFiles',strcat(exp,'_pos.jpg'));
%     figname12 = fullfile(pwd,'experiments',exp,'outputFiles',strcat(exp,'_pos_eps.eps'));
% 
%     if filesSaved == 1
%         exportgraphics(f1,figname11)
%         exportgraphics(f1,figname12)
%     end
% end
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% if figure2_process == 1
%     bSe1_htm = select(bag,'Topic','/feats_htm');
%     msg_htm = readMessages(bSe1_htm,'DataFormat','struct');
%     D1 = cellfun(@(m) find(m.Data),msg_htm,'UniformOutput',false);
%     S = evaluateSim(D1(offset1:size(msg_htm,1)-offset2),D1(offset1:size(msg_htm,1)-offset2),'wincell');
%     
%     figure(2)
%     s = imagesc(S);
%     colorbar
%     title('Confusion Matrix')
%     xlabel('Query images')
%     ylabel('Database images')
%     
% %     size_gt = size(S);
% %     GT = eye(size_gt(1,1));
% %     tole = 10;
% %     rep = 3;
% %     vpr = 120;
% %     
% %     for j = 1:4
% %         for i=1:size_gt(1,1)
% %             pos_gt = i + (j-1)*vpr;
% %             if (pos_gt+tole)>size_gt(1,1)
% %                 break
% %             end
% %             GT(i,pos_gt:(pos_gt+tole))=1;
% %         end
% %     end
%     
%     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     if plotConfig == 1
%         width = 4;      % Width in inches
%         height = 3;     % Height in inches
%         alw = 0.75;     % AxesLineWidth
%         fsz = 9;        % Fontsize
%         lw = 1.5;       % LineWidth
%         msz = 8;        % MarkerSize
%     
%         f2 = figure(2);
%         pos = get(f2, 'Position');
%         ff2 = get(f2,'Children');
%     
%         set(f2, 'Position', [pos(1) pos(2) width*100, height*100]);
%         set(ff2, 'FontSize', fsz, 'LineWidth', alw); %<- Set properties
%     
%         figname21 = fullfile(pwd,'experiments',exp,'outputFiles',strcat(exp,'_conf.jpg'));
%         figname22 = fullfile(pwd,'experiments',exp,'outputFiles',strcat(exp,'_conf_eps.eps'));
%     
%         if filesSaved == 1
%             exportgraphics(f2,figname21)
%             exportgraphics(f2,figname22)
%         end
%     end
% end
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% bSel_vt = select(bag,'Topic','/LocalView/Template');
% msg_vt = readMessages(bSel_vt,'DataFormat','struct');
% id = cellfun(@(m) double(m.CurrentId),msg_vt);
% 
% figure(3)
% plot(id, 'bx')
% grid on;
% grid minor
% %stem(id)
% title('View Cells created')
% xlabel('Query images')
% ylabel('Id View Cells')
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% if plotConfig == 1
%     width = 3;      % Width in inches
%     height = 3;     % Height in inches
%     alw = 0.75;     % AxesLineWidth
%     fsz = 9;        % Fontsize
%     lw = 1.5;       % LineWidth
%     msz = 8;        % MarkerSize
% 
%     f3 = figure(3);
%     pos = get(f3, 'Position');
%     ff3 = get(f3,'Children');
% 
%     set(f3, 'Position', [pos(1) pos(2) width*100, height*100]);
%     set(ff3, 'FontSize', fsz, 'LineWidth', alw); %<- Set properties
% 
%     figname31 = fullfile(pwd,'experiments',exp,'outputFiles',strcat(exp,'_vc.jpg'));
%     figname32 = fullfile(pwd,'experiments',exp,'outputFiles',strcat(exp,'_vc_eps.eps'));
% 
%     if filesSaved == 1
%         exportgraphics(f3,figname31)
%         exportgraphics(f3,figname32)
%     end
% end
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% data_em = select(bag,'Topic','/ExperienceMap/Map');
% msg = readMessages(data_em,'DataFormat','struct');
% em = msg{data_em.AvailableTopics.NumMessages};
%  
% x = em.Node(:);
% n = em.NodeCount;
% y0 = double(vertcat(x.Id));
% y1 = vertcat(x.Pose)';
% z1 = vertcat(y1.Position);
% z2 = vertcat(y1.Orientation);
% w = vertcat([z1.X; z1.Y; z1.Z; z2.X; z2.Y; z2.Z; z2.W])';
% pose = [y0 w];
% 
% x = em.Edge(:);
% nl = em.EdgeCount;
% y0 = vertcat(x.Id);
% y1 = vertcat(x.SourceId);
% y2 = vertcat(x.DestinationId);
% links = [y0 y1 y2];
% 
% markersize = 5;
% linewidth = 2;
% draw_links = 1;
% 
% %pose_em(:,2) = pose(offset1:end-offset2, 2)-pose(1,2);
% %pose_em(:,3) = pose(offset1:end-offset2, 3)-pose(1,3);
% 
% pose_em(:,2) = pose(offset1:end-offset2, 2);
% pose_em(:,3) = pose(offset1:end-offset2, 3);
% 
% f4 = figure(4);
% %subplot(1,2,2);
% plot(pose_em(:,2), pose_em(:,3), 'go', 'MarkerSize', markersize, 'MarkerFaceColor', 'none', 'LineWidth', linewidth)
% %axis([-15 5 -4 14])
% %hold on
% %plot(pose_odom_frame(:,1)-pose_odom_frame(1,1),pose_odom_frame(:,2)-pose_odom_frame(1,2));
% %axis equal;
% title('Map Experience')
% 
% if draw_links == 1
% 	if nl > 0
%         for k = offset1:nl
%             sn = links(k, 2) + 1;
%             dn = links(k, 3) + 1;
%             line([pose(sn, 2) pose(dn, 2)], [pose(sn, 3) pose(dn, 3)]);                
%         end
% 	end
% end
% 
% xlabel('x(m)');
% ylabel('y(m)');
% grid on;
% grid minor
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% if plotConfig == 1
%     width = 3;      % Width in inches
%     height = 3;     % Height in inches
%     alw = 0.75;     % AxesLineWidth
%     fsz = 9;        % Fontsize
%     lw = 1.5;       % LineWidth
%     msz = 8;        % MarkerSize
% 
%     pos = get(f4, 'Position');
%     ff4 = get(f4,'Children');
% 
%     set(f4, 'Position', [pos(1) pos(2) width*100, height*100]);
%     set(ff4, 'FontSize', fsz, 'LineWidth', alw); %<- Set properties
% 
%     figname41 = fullfile(pwd,'experiments',exp,'outputFiles',strcat(exp,'_em.jpg'));
%     figname42 = fullfile(pwd,'experiments',exp,'outputFiles',strcat(exp,'_em_eps.eps'));
% 
%     if filesSaved == 1
%         exportgraphics(f4,figname41)
%         exportgraphics(f4,figname42)
%     end
% end
% 
% % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % width = 3;      % Width in inches
% % height = 3;     % Height in inches
% % alw = 0.75;     % AxesLineWidth
% % fsz = 9;        % Fontsize
% % lw = 1.5;       % LineWidth
% % msz = 8;        % MarkerSize
% % 
% % pos = get(f1, 'Position');
% % ff1 = get(f1,'Children');
% % 
% % set(f1, 'Position', [pos(1) pos(2) width*100, height*100]);
% % set(ff1, 'FontSize', fsz, 'LineWidth', alw); %<- Set properties
% % 
% % fileName = fullfile(pwd,'outputFiles', exp);
% % figname11 = [strcat(fileName,'\',exp,'_em') '.jpg'];
% % figname12 = [strcat(fileName,'\',exp,'_em_eps') '.eps'];
% % 
% % if filesSaved == 1
% % 	exportgraphics(f1,figname11)
% % 	exportgraphics(f1,figname12)
% % end
% % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % data_em_pose = select(bag,'Topic','/ExperienceMap/RobotPose');
% % msg_em = readMessages(data_em_pose,'DataFormat','struct');
% % x_em_pose = cellfun(@(m) double(m.Pose.Position.X),msg_em);
% % y_em_pose = cellfun(@(m) double(m.Pose.Position.Y),msg_em);
% % 
% % f5 = figure(5);
% % plot(x_em_pose(:,1),y_em_pose(:,1));
% % hold on
% % plot(pose_odom_frame(:,1),pose_odom_frame(:,2));
% % title('Husky Position Odometry')
% % xlabel('x(m)')
% % ylabel('y(m)')
% % grid on
% % grid minor
% % 
% % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % data_info = select(bag,'Topic','/info');
% % info = readMessages(data_info,'DataFormat','struct');
% % info_id = cellfun(@(m) uint32(m.CurrentViewCell),info,'UniformOutput',false);
% % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % figure(5)
% % GT = createGT(size_gt(1,2),0,2,[135,245,365]);
% % [P, R, F1] = createPR(S(10:end), GT(10:end));
% % plot(R,P);
% % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % Tests with Confusion Matrix using RatSlam. 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% if ratslam_var == 1
%     features = cellfun(@(m) double(m.Feature),msg_vt,'UniformOutput',false);
%     feature_max = max(cell2mat(features));
%     features_values = size(features);
%     features_values = features_values(1,1);
%     ratslam_matrix = eye(features_values);
%     for i=1:(features_values-1)
%         aux = cell2mat(features(i+1));
%         aux_size = size(aux);
%         aux_size = aux_size(1,1);
%         for j=1:i
%             if j <= aux_size
%                 ratslam_matrix(j,i+1) = (feature_max - aux(j))/feature_max;
%             end
%         end
%     end
%     figure(6);
%     s = imagesc(ratslam_matrix);
%     colorbar
%     title('Confusion Matrix')
%     xlabel('Query images')
%     ylabel('Database images')
% 
%     vc_created = max(id) + 2;
%     ratslam_vc_matrix = eye(vc_created);
%     aux_size_prev = 0;
%     for i=1:features_values
%         aux = cell2mat(features(i));
%         aux_size = size(aux);
%         aux_size = aux_size(1,1);
%     
%         if aux_size == aux_size_prev 
%             continue
%         end
%     
%         for j=1:aux_size
%             ratslam_vc_matrix(j,aux_size+1) = (feature_max - aux(j))/feature_max;
%         end
%     
%         aux_size_prev = aux_size;
%     end
% 
%     figure(7);
%     s = imagesc((ratslam_vc_matrix + ratslam_vc_matrix')/2);
%     colorbar
%     title('Confusion Matrix')
%     xlabel('Query View Cells')
%     ylabel('Database View Cells')
% 
%     figure(8)
%     GT = createGT(vc_created,0,20,[620,1020,1420]);
%     s = imagesc(GT);
%     colorbar
%     title('Confusion Matrix')
%     xlabel('Query View Cells')
%     ylabel('Database View Cells')
% 
%     figure(9)
%     [P, R, F1] = createPR((ratslam_vc_matrix + ratslam_vc_matrix')/2, GT);
%     plot(R,P);
% end
% 
