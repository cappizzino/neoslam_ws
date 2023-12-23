close all
clearvars

filesSaved = 1;
plotConfig = 1;

neoslam_var = 1;
ratslam_var = 1;

%% Neoslam


if neoslam_var == 1
    file = fullfile('experiments','robotarium_neoslam','_2023-09-23-18-38-24_0.bag');
    bag = rosbag(file);
    
    bSe1_htm = select(bag,'Topic','/feats_htm');
    msg_htm = readMessages(bSe1_htm,'DataFormat','struct');
    D1 = cellfun(@(m) find(m.Data),msg_htm,'UniformOutput',false);
    S_htm = evaluateSim(D1(1:end),D1(1:end),'wincell');
    
%     figure(1)
%     s = imagesc(S_htm);
%     colorbar
%     title('Confusion Matrix')
%     xlabel('Query images')
%     ylabel('Database images')

    size_gt = size(S_htm);
    
    %% CNN
    bSe1_cnn = select(bag,'Topic','/feats_cnn');
    msg_cnn = readMessages(bSe1_cnn,'DataFormat','struct');
    CNN1 = cellfun(@(m) double(m.Data), msg_cnn,'UniformOutput',false);
    
    cnn = zeros(length(CNN1),length(CNN1{1}));
    for i = 1:length(CNN1)
        cnn(i,:) = CNN1{i}';
    end
    S_cnn = evaluateSim(cnn,cnn,'cosine');
    
%     figure(2)
%     s = imagesc(S_cnn);
%     colorbar
%     title('Confusion Matrix')
%     xlabel('Query images')
%     ylabel('Database images')
%     
    %% Ground truth
    GT = createGT(size_gt(1,2),0,2,[135,245,365]);
    
    %% Precision-recall calculations
    [P_htm, R_htm, F1_htm] = createPR(S_htm(10:end), GT(10:end));
    mF1_htm = max(F1_htm);
    AP_htm = trapz(R_htm, P_htm);

    [P_cnn, R_cnn, F1_cnn] = createPR(S_cnn(10:end), GT(10:end));
    mF1_cnn = max(F1_cnn);
    AP_cnn = trapz(R_cnn, P_cnn);
    
%     figure(3)
%     plot(R_neo,P_neo);
end

%% RatSlam
if ratslam_var == 1
    file = fullfile('experiments','robotarium_ratslam','ratslam_robotarium_features.bag');
    bag = rosbag(file);

    bSel_vt = select(bag,'Topic','/LocalView/Template');
    msg_vt = readMessages(bSel_vt,'DataFormat','struct');
    id = cellfun(@(m) double(m.CurrentId),msg_vt);
    
%     figure(4)
%     plot(id, 'bx')
%     grid on;
%     grid minor
%     %stem(id)
%     title('View Cells created')
%     xlabel('Query images')
%     ylabel('Id View Cells')

    features = cellfun(@(m) double(m.Feature),msg_vt,'UniformOutput',false);
    feature_max = max(cell2mat(features));
    features_values = size(features);
    features_values = features_values(1,1);
    ratslam_matrix = eye(features_values);
    for i=1:(features_values-1)
        aux = cell2mat(features(i+1));
        aux_size = size(aux);
        aux_size = aux_size(1,1);
        for j=1:i
            if j <= aux_size
                ratslam_matrix(j,i+1) = (feature_max - aux(j))/feature_max;
            end
        end
    end
%     figure(5);
%     s = imagesc(ratslam_matrix);
%     colorbar
%     title('Confusion Matrix')
%     xlabel('Query images')
%     ylabel('Database images')

    vc_created = max(id) + 2;
    ratslam_vc_matrix = eye(vc_created);
    aux_size_prev = 0;
    for i=1:features_values
        aux = cell2mat(features(i));
        aux_size = size(aux);
        aux_size = aux_size(1,1);
    
        if aux_size == aux_size_prev 
            continue
        end
    
        for j=1:aux_size
            ratslam_vc_matrix(j,aux_size+1) = (feature_max - aux(j))/feature_max;
        end
    
        aux_size_prev = aux_size;
    end

%     figure(6);
%     s = imagesc((ratslam_vc_matrix + ratslam_vc_matrix')/2);
%     colorbar
%     title('Confusion Matrix')
%     xlabel('Query View Cells')
%     ylabel('Database View Cells')
    GT = createGT(vc_created,0,20,[620,1020,1420]);

%     figure(7)
%     s = imagesc(GT);
%     colorbar
%     title('Confusion Matrix')
%     xlabel('Query View Cells')
%     ylabel('Database View Cells')

    [P, R, F1] = createPR((ratslam_vc_matrix + ratslam_vc_matrix')/2, GT);
    mF1 = max(F1);
    AP = trapz(R, P);
    
%     figure(8)
%     plot(R,P);
end

%% Plots
style = ["-", "--", ":"]; 
color = ["r", "g", "b", "c", "m"]; 

figure(9);

plot(R, P, char(strcat(style(1),color(1))));
legends{1} = ['RatSlam (AUC=', sprintf('%.2f',AP), ')'];
hold on

plot(R_cnn, P_cnn, char(strcat(style(1),color(4))));
legends{2} = ['CNN (AUC=', sprintf('%.2f',AP_cnn), ')'];
hold on

plot(R_htm, P_htm, char(strcat(style(1),color(3))));
legends{3} = ['HTM (AUC=', sprintf('%.2f',AP_htm), ')'];
hold on

hold off
legend(legends, 'location', 'northeast'); %southwest northeast
xlabel('Recall');
ylabel('Precision');
grid('on');
title('PR curves');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if plotConfig == 1
    width = 4;      % Width in inches
    height = 3;     % Height in inches
    alw = 0.75;     % AxesLineWidth
    fsz = 8;        % Fontsize
    lw = 1.5;       % LineWidth
    msz = 8;        % MarkerSize

    f9 = figure(9);
    pos = get(f9, 'Position');
    ff9 = get(f9,'Children');

    set(f9, 'Position', [pos(1) pos(2) width*100, height*100]);
    set(ff9, 'FontSize', fsz, 'LineWidth', alw); %<- Set properties

    figname91 = fullfile(pwd,strcat('_robotarium_pr.jpg'));
    figname92 = fullfile(pwd,strcat('_robotarium_pr.eps'));

    if filesSaved == 1
        exportgraphics(f9,figname91)
        exportgraphics(f9,figname92)
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure(10);

x = categorical({'Maximum F1 score'});
vals = [mF1 ; mF1_cnn ; mF1_htm];
b = bar(x,vals);
ylim([0 0.4])

xtips1 = b(1).XEndPoints;
ytips1 = b(1).YEndPoints;
labels1 = sprintf('%.2f',b(1).YData);
text(xtips1,ytips1,labels1,'HorizontalAlignment','center',...
    'VerticalAlignment','bottom','FontSize', 8)

xtips2 = b(2).XEndPoints;
ytips2 = b(2).YEndPoints;
labels2 = sprintf('%.2f',b(2).YData);
text(xtips2,ytips2,labels2,'HorizontalAlignment','center',...
    'VerticalAlignment','bottom', 'FontSize', 8)

xtips3 = b(3).XEndPoints;
ytips3 = b(3).YEndPoints;
labels3 = sprintf('%.2f',b(3).YData);
text(xtips3,ytips3,labels3,'HorizontalAlignment','center',...
    'VerticalAlignment','bottom', 'FontSize', 8)

legend('RatSlam','CNN','HTM','Location','northwest')

hold on

ylabel('F1 Score');
grid('on');
title('Maximum F1 score');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if plotConfig == 1
    width = 4;      % Width in inches
    height = 3;     % Height in inches
    alw = 0.75;     % AxesLineWidth
    fsz = 8;        % Fontsize
    lw = 1.5;       % LineWidth
    msz = 8;        % MarkerSize

    f10 = figure(10);
    pos = get(f10, 'Position');
    ff10 = get(f10,'Children');

    set(f10, 'Position', [pos(1) pos(2) width*100, height*100]);
    set(ff10, 'FontSize', fsz, 'LineWidth', alw); %<- Set properties

    figname101 = fullfile(pwd,strcat('_robotarium_f1.jpg'));
    figname102 = fullfile(pwd,strcat('_robotarium_f1.eps'));

    if filesSaved == 1
        exportgraphics(f10,figname101)
        exportgraphics(f10,figname102)
    end
end
