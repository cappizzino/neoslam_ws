function [C] = select_vp(actions_vp)
% actions_vp =[[2, 3, 5];
%              [2, 4, 6];
%              [2, 4, 7];
%              [2, 8, 5]];

vpr_merge = {};
places_number = size(actions_vp,1);

for i=1:places_number
    source_id = actions_vp(i,2);
    [row,col] = find(actions_vp(:,2)==source_id);
    vp1 = actions_vp(row,3)';
    [row,col] = find(actions_vp(:,3)==vp1);
    vp2 = actions_vp(row,2)';
    vpr_merge{i}=[source_id, vp1, vp2];
    %pause()
end

B = cellfun(@(v) sort(v),vpr_merge,'uni',0);
C = cellfun(@(v) unique(v), B,'uni',0);