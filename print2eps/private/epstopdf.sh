FILES="*.eps"
for f in $FILES; do ps2pdf -dEPSCrop $f; done
