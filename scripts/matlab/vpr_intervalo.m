close all
clearvars

exp = 'outdoor_afternoon'; % 'robotarium'; % 'corridor' %'outdoor_afternoon'

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
switch exp
    case 'robotarium'
        file =  'E:\dataset\output\robotarium\husky_out_cnn.bag';
        offset1 = 10;
        offset2 = 20;
        theta = -58;%-56;
    case 'corridor'
        file = 'C:\dataset\_2022-04-07-11-08-05_corridor.bag';
        offset1 = 10;
        offset2 = 20;
        theta = 0;
    case 'outdoor_afternoon'
        file = 'E:\dataset\output\outdoor_afternoon\husky_out.bag';
        file2 = 'E:\dataset\output\outdoor_morning\husky_out.bag';
    otherwise
        warning('No plot created.')
        return
end

bag = rosbag(file);
bag2 = rosbag(file2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nCols = 2048;
nCellsCol = 32;
nFeatures = 13*13*384;

bSe1_cnn = select(bag,'Topic','/feats_cnn');
msg_cnn = readMessages(bSe1_cnn,'DataFormat','struct');
Dcnn = cellfun(@(m) double(m.Data),msg_cnn,'UniformOutput',false);

%nCells = nCols*nCellsCol;
%D = zeros(nCellsCol,nCols,'single');
D = zeros(length(Dcnn),nFeatures,'single');

for i = 1:length(Dcnn)
	D(i,:) = Dcnn{i}';
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
bSe1_htm = select(bag,'Topic','/feats_htm');
msg_htm = readMessages(bSe1_htm,'DataFormat','struct');
D_htm = cellfun(@(m) find(m.Data),msg_htm,'UniformOutput',false);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
D_htm_sparse = logical(sparse(length(D_htm), 2^16));
obs = 1;
n_interval = 1;
theta_alpha = 400;
theta_rho = 2;
interval = struct('InitEnd',{},'anchor',[],'descriptors',[],'global',[]);

for i = 1:length(D_htm)
    D_htm_sparse(i, D_htm{i}) = 1;
    
    if obs == 1
        interval(n_interval).InitEnd = [1,1];
        interval(n_interval).anchor = D_htm_sparse(obs,:);
        interval(n_interval).descriptors = D_htm_sparse(obs,:);
        interval(n_interval).global = D_htm_sparse(obs,:);
    else
        alpha = interval(n_interval).anchor * D_htm_sparse(obs,:)';
        o_distance = interval(n_interval).InitEnd(1,2) - interval(n_interval).InitEnd(1,1);
        
        if (alpha >= theta_alpha) && (o_distance < theta_rho)
                interval(n_interval).InitEnd(1,2) = interval(n_interval).InitEnd(1,2) + 1;
                interval(n_interval).descriptors = logical([interval(n_interval).descriptors;...
                D_htm_sparse(obs,:)]);
                interval(n_interval).global = or(interval(n_interval).global,D_htm_sparse(obs,:));
        else
            n_interval = n_interval + 1;
            interval(n_interval).InitEnd = [obs,obs];
            interval(n_interval).anchor = D_htm_sparse(obs,:);
            interval(n_interval).descriptors = D_htm_sparse(obs,:);
            interval(n_interval).global = D_htm_sparse(obs,:);
        end
    end
    image_interval(obs) = n_interval;
    obs = obs + 1;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
j = 300;
k = 4;

D_test = D_htm_sparse(j, :);

for i = 1:n_interval
    simil(i) = sum(xor(interval(i).global, D_test));
end
%bar(simil)
%b = full(simil(1,:));
%b = b(1,2:end);
%[c,ci] = sort(b);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
bSe1_htm2 = select(bag2,'Topic','/feats_htm');
msg_htm2 = readMessages(bSe1_htm2,'DataFormat','struct');
D_htm2 = cellfun(@(m) find(m.Data),msg_htm2,'UniformOutput',false);
D_htm_sparse2 = logical(sparse(length(D_htm2), 2^16));

for i = 1:length(D_htm2)
    D_htm_sparse2(i, D_htm2{i}) = 1;
end

j = 300;
D_test2 = D_htm_sparse2(j, :);

for i = 1:n_interval
    simil2(i) = sum(xor(interval(i).global, D_test2));
end
bar(simil2)
b = full(simil2(1,:));
b = b(1,2:end);
[c,ci] = sort(b);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% nImage = 20;
% 
% for i = 1:length(Dcnn)
%     xy = dot(D(nImage,:),D(i,:));
%     nx   = norm(D(nImage,:));
%     ny   = norm(D(i,:));
%     nxny = nx*ny;
%     overlap(i) = xy/nxny;
% end

% figure(1)
% bar(overlap)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%