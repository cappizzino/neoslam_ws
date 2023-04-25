% Calculate the similarities between vectors.
function [S] = evaluateSim (D1, D2, method)

switch method
    case 'wincell'
        max_cell_id_D1 = uint64(max(cellfun(@max, D1)));
        max_cell_id_D2 = uint64(max(cellfun(@max, D2)));
        max_cell_id = uint64(max(max_cell_id_D1, max_cell_id_D2));
    
        D2_sparse = sparse(length(D2), max_cell_id);
        for i = 1:length(D2)
            D2_sparse(i, D2{i}) = 1;
        end
        D2 = D2_sparse;
    
        D1_sparse = sparse(length(D1), max_cell_id);
        for i = 1:length(D1)
            D1_sparse(i, D1{i}) = 1;
        end
        D1 = D1_sparse;
    
        S = D1 * D2';
        nOnes_D1 = sum(D1, 2);
        nOnes_D2 = sum(D2, 2);
        mean_nOnes = [ones(length(nOnes_D1), 1), nOnes_D1] * [nOnes_D2'; ones(1, length(nOnes_D2))] / 2;
    
        S = S ./ mean_nOnes;
        S = full(S);
        
    case 'cosine'
        S = 1 - pdist2(D1, D2, method);
end
  
end