function adjacent_points = computeAdjacentPoints(i,j,k,adjacent_distance,dx,dy,dz)
    % adjacent_points = [];
    % for ii=i-adjacent_distance:i+adjacent_distance
    %     rou = round(sqrt(adjacent_distance*adjacent_distance - (ii-i)*(ii-i)));
    %     for jj=j-rou:j+rou
    %         rest = round(sqrt(rou*rou - (jj-j)*(jj-j)));
    %         if rest==0
    %             kk=k;
    %             adjacent_points = [adjacent_points;sub2ind([dx,dy,dz], ii, jj, kk)];
    %         else
    %             kk=k-rest;
    %             adjacent_points = [adjacent_points;sub2ind([dx,dy,dz], ii, jj, kk)];
    %             kk=k+rest;
    %             adjacent_points = [adjacent_points;sub2ind([dx,dy,dz], ii, jj, kk)];
    %         end
    %     end
    % end

    adjacent_points = zeros(6,1);
    temp = sub2ind([dx,dy,dz], i-adjacent_distance, j, k);
    adjacent_points(1) = temp;
    temp = sub2ind([dx,dy,dz], i+adjacent_distance, j, k);
    adjacent_points(2) = temp;
    temp = sub2ind([dx,dy,dz], i, j-adjacent_distance, k);
    adjacent_points(3) = temp;
    temp = sub2ind([dx,dy,dz], i, j+adjacent_distance, k);
    adjacent_points(4) = temp;
    temp = sub2ind([dx,dy,dz], i, j, k-adjacent_distance);
    adjacent_points(5) = temp;
    temp = sub2ind([dx,dy,dz], i, j, k+adjacent_distance);
    adjacent_points(6) = temp;