function newmap = bwr(m)
%BLUEWHITERED   Blue, white, and red color map.
%   BLUEWHITERED(M) returns an M-by-3 matrix containing a blue to white
%   to red colormap, with white corresponding to the CAXIS value closest
%   to zero.  This colormap is most useful for images and surface plots
%   with positive and negative values.  BLUEWHITERED, by itself, is the
%   same length as the current colormap.
%
%   Examples:
%   ------------------------------
%   figure
%   imagesc(peaks(250));
%   colormap(bluewhitered(256)), colorbar
%
%   figure
%   imagesc(peaks(250), [0 8])
%   colormap(bluewhitered), colorbar
%
%   figure
%   imagesc(peaks(250), [-6 0])
%   colormap(bluewhitered), colorbar
%
%   figure
%   surf(peaks)
%   colormap(bluewhitered)
%   axis tight
%
%   See also HSV, HOT, COOL, BONE, COPPER, PINK, FLAG, 
%   COLORMAP, RGBPLOT.


if nargin < 1
   m = size(get(gcf,'colormap'),1);
end


bottom = [0 0 1];
%botmiddle = [0 0 0.7];
middle = [1 1 1];
%topmiddle = [0.7 0 0];
top = [1 0 0];

% Find middle
lims = get(gca, 'CLim');

    ratio = abs(lims(2)) / (abs(lims(1)) + lims(2));
    poslen = round(m*ratio)+0;
    neglen = m - poslen;
    maxlen = max(neglen,poslen);

    newmap_neg = zeros(neglen, 3);
     
    for i=1:3
        if (i==3)
        newmap_neg(:,i) = 1;
        end
        if (i~=3)
            for j=1:1:neglen
                newmap_neg(j,i) = ((maxlen - neglen + j)/maxlen);
            end
        end
    end
    
    newmap_pos = zeros(poslen, 3);

    for i=1:3
         if (i==1)
            newmap_pos(:,i) = 1;
         end
         if (i~=1)
            for j=1:1:poslen
                newmap_pos(j,i) = ((maxlen - j + 1)/maxlen);
            end
         end
    end
    
    newmap = [newmap_neg; newmap_pos];
