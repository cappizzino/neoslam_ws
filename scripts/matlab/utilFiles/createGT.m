function GT = createGT(size_gt, offset, alpha, points)

vector = [1, points];
vector_size = size(vector);
vector_size = vector_size(1,2);

for j = 1:vector_size
    for i=1:(size_gt)
        pos_gt = i + vector(j) - 1;
        for w = (pos_gt - alpha/2):(pos_gt + alpha/2)
            if (w > 0) && (w < size_gt+1)
                GT(i, w) = 1;
            end
        end
    end

    for i=1:(size_gt)
        pos_gt = i - vector(j) + 1;
        for w = (pos_gt - alpha/2):(pos_gt + alpha/2)
            if (w > 0) && (w < size_gt+1)
                GT(i, w) = 1;
            end
        end
    end
end

GT = GT((offset+1):size_gt,(offset+1):size_gt);