%
%   =====================================================================
%   Copyright (C) 2019  
%   Peer Neubert, peer.neubert@etit.tu-chemnitz.de
%   Stefan Schubert, stefan.schubert@etit.tu-chemnitz.de
%
%   This program is free software: you can redistribute it and/or modify
%   it under the terms of the GNU General Public License as published by
%   the Free Software Foundation, either version 3 of the License, or
%   (at your option) any later version.
%   
%   This program is distributed in the hope that it will be useful,
%   but WITHOUT ANY WARRANTY; without even the implied warranty of
%   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%   GNU General Public License for more details.
%   
%   You should have received a copy of the GNU General Public License
%   along with this program.  If not, see <http://www.gnu.org/licenses/>.
%   =====================================================================
%
% Software release for the paper:
% Neubert, P., Schubert, S. & Protzel, P. (2019) A neurologically inspired 
% sequence processing model for mobile robot place recognition. In IEEE 
% Robotics and Automation Letters (RA-L) and presentation at Intl. Conf. 
% on Intelligent Robots and Systems (IROS). DOI: 10.1109/LRA.2019.2927096 
%
%
% Computes precision and recall vectors for a given similarrity matrix
% and a binary ground truth matrix. 
%
% S ... similarity matrix
% GThard ... ground truth matching matrix: 1 at places that must be matched,
%            else 0
%
function [P, R, F1] = createPR(S, GThard)

    GT = logical(GThard); % ensure logical-datatype
    
    % init precision and recall vectors
    R = 0;
    P = 1;
    F1 = 0;
    
    % select start and end treshold
    startV = max(S(:)); % start-value for treshold
    endV = min(S(:)); % end-value for treshold
    
    % iterate over different thresholds
    for i=linspace(startV, endV, 100) 
        B = S>=i; % apply threshold
        
        TP = nnz( GT & B ); % true positives
        FN = nnz( GT & (~B) ); % false negatives
        FP = nnz( (~GT) & B ); % false positives
        
        precision = TP/(TP + FP);
        recall = TP/(TP + FN);
        
        P(end+1) = precision;
        R(end+1) = recall;
        F1(end+1) = (2 * precision * recall) / (precision + recall);
    end
end
