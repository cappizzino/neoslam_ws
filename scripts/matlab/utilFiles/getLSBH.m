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
% Create a sLSBH from desciptors and projection matrix.
%
% Y ... descriptors as m-by-n matrix with m descriptors and n features
% P ... Projection matrix; e.g. P = normc(randn(8192, 16384, 'single'));
% s ... sparsity with s = (0,1]
%
function L = getLSBH(Y, P, s)

  n = round(size(P,2)*s);

  % random projection
  Y2 = Y * P;
  
  % sort
  [~, IDX] = sort(Y2,2, 'descend');
  
  % sparsification
  L1 = zeros(size(Y2), 'logical');
  
  for i = 1:size(Y2,1)
    L1(i,IDX(i,1:n)) = 1;
  end
  
  % sort
  [~, IDX] = sort(Y2,2, 'ascend');
  
  % sparsification
  L2 = zeros(size(Y2), 'logical');
  
  for i = 1:size(Y2,1)
    L2(i,IDX(i,1:n)) = 1;
  end
  
  % concat
  L = single([L1, L2]);
end


