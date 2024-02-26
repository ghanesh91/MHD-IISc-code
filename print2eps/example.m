% imsavecircles example

image = imread('coins.png');
image = image(20:end-30,30:end-20);
filename = 'circles.png';
radiiRange = [15 30];
margin = max(radiiRange);
[centers, radii, metric] = imfindcircles(image,radiiRange);
imsavecircles(image,centers,radii,margin,filename)