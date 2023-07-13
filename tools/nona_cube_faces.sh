#!/bin/sh
echo "Bash scripts to generate cubic faces for equirectangular panorama image"
if [ $# != 6 ];then
    echo "Usage: $0"
    echo "          src_file   #path to full spherical panorama image"
    echo "          cube_dim   #cube dimension in pixels"
    echo "          cube_fov   #cube field of vision (default 90 degree)"
    echo "          width      #width of source image in pixels"
    echo "          height     #height of source image in pixels[should be 1/2 * width]"
    echo "          output_dir #output directory for cubic map"
    exit -1
fi
echo "executed command: $0 $1 $2 $3 $4 $5 $6"
suffix=JPG
nona_exe=`command -v nona`
if [ $? -ne 0 ]; then
    echo "nona does not exist,install it now ..."
    read -s -p "Enter Password for installing nona tools from hugin: " passwd
    echo $passwd | sudo -S apt-get install hugin hugin-tools hugin-data
fi
echo "Use nona tool: $nona_exe"

# faces are "$1"-{left,right,front,back,up,down}.jpg
# p-line describes rectilinear panorama on cube face
p="p f0 w$2 h$2 v$3"

# m-line: gamma 1.0 spline36 interpolator
m="m g1 i2"

# o-line selects from spherical full panorama
o="o f4 w$4 h$5 v360 r0"

output_dir=$6
# temporary file for script
tmp="$output_dir/$$.oto"

# remove oto file when shell exits
trap 'rm -f $tmp' 0
# remove previous faces, if any

cd $output_dir
echo "first param: $1"
spot_name=$(basename $1 .${suffix})
#rm -f "$spot_name"-*.${suffix}

# create scripts and extract the front faces
cat > "$tmp" << eof
$p
$m
$o p0 y0 n"$1"
eof
nona -m JPEG -z 95 -o "$spot_name"-front.${suffix} "$tmp"

# create scripts and extract the right faces
cat > "$tmp" << eof
$p
$m
$o p0 y270 n"$1"
eof
nona -m JPEG -z 95 -o "$spot_name"-right.${suffix} "$tmp"

# create scripts and extract the back faces
cat > "$tmp" << eof
$p
$m
$o p0 y180 n"$1"
eof
nona -m JPEG -z 95 -o "$spot_name"-back.${suffix} "$tmp"

# create scripts and extract the left faces
cat > "$tmp" << eof
$p
$m
$o p0 y90 n"$1"
eof
nona -m JPEG -z 95 -o "$spot_name"-left.${suffix} "$tmp"

# create scripts and extract the up faces
cat > "$tmp" << eof
$p
$m
$o p270 y0 n"$1"
eof
nona -m JPEG -z 95 -o "$spot_name"-up.${suffix} "$tmp"

# create scripts and extract the down faces
cat > "$tmp" << eof
$p
$m
$o p90 y0 n"$1"
eof
nona -m JPEG -z 95 -o "$spot_name"-down.${suffix} "$tmp"
exit 0
