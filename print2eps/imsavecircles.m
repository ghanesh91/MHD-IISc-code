  function imsavecircles(image,centers,radii,margin,filename,varargin)
        %IMSAVECIRCLES plots and saves result of imfindcircles with
        %   transparent margin
        %
        %   IMSAVECIRCLES(im,centers,radii,margin,filename) displays im
        %   with the circles specified by centers and radii (see
        %   imfindcircles). A transparent margin (integer) is added to show
        %   circles at the edge of the image. To get reliable results, use
        %   the maximum possible circle radius for this. The image is then
        %   saved at native resolution to a file specified by filename using
        %   export_fig. Save as a .png to keep transparency in the margin
        %
        %   IMSAVECIRCLES(im,centers,radii,margin,filename,format) allows
        %   to specify formatting of circles as
        %   {innerColor,innerStyle,innerWidth,outerColor,outerStyle,outerWidth}
        %   Default is {'b','-',2,'w','-',3}. For Poker Chips, use {'b','--',2,'w','-',3}

        
        narginchk(5,6)
        if nargin == 6
            format = varargin{1};
        else
            format = {'b','-',2,'w','-',3};
        end
        
        innerColor = format{1};
        innerStyle = format{2};
        innerWidth = format{3};
        outerColor = format{4};
        outerStyle = format{5};
        outerWidth = format{6};
        
        % create alpha channel to make margin transparent
        alpha = ones(size(image));
        
        % add margin: pad image and alpha channel with zeros (=transparent)
        image = padarray(image,margin*[1 1]);
        alpha = padarray(alpha,margin*[1 1]);
        
        % display image and circles
        figure
        hImage = imshow(image,'Border', 'tight');
        hCircles = viscircles(centers+margin, radii);
        
        % use alpha channel to make border transparent
        set(hImage,'AlphaData',alpha);
        
        % set circle properties
        hCircles = get(hCircles,'Children');
        set(hCircles(1),'Color',innerColor)
        set(hCircles(1),'LineStyle',innerStyle)
        set(hCircles(1),'LineWidth',innerWidth)
        
        set(hCircles(2),'Color',outerColor)
        set(hCircles(2),'LineStyle',outerStyle)
        set(hCircles(2),'LineWidth',outerWidth)
        
        % export at native resolution preserving the transparent margin
        export_fig(filename,'-native','-transparent','-nocrop')
    end