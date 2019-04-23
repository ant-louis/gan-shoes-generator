#!bin/bash

dest=/home/tom/Documents/Uliege/SecondSemester/Deep\ Learning/Shoes-generator/dataset/
source=/home/tom/Documents/Uliege/SecondSemester/Deep\ Learning/Shoes-generator/ut-zap50k-images-square/


source_subdir=Boots/Ankle	
dest_subdir=boots-ankle
cd "$source"/"$source_subdir"
find . -type f -print0 | xargs -0 mv -t "$dest"/"$dest_subdir" 

source_subdir=Boots/Knee\ High	
dest_subdir=boots-kneehigh
cd "$source"/"$source_subdir"
find . -type f -print0 | xargs -0 mv -t "$dest"/"$dest_subdir" 

source_subdir=Boots/Mid-Calf	
dest_subdir=boots-midcalf
cd "$source"/"$source_subdir"
find . -type f -print0 | xargs -0 mv -t "$dest"/"$dest_subdir" 

source_subdir=Sandals/Flat	
dest_subdir=sandals-flats
cd "$source"/"$source_subdir"
find . -type f -print0 | xargs -0 mv -t "$dest"/"$dest_subdir" 

source_subdir=Shoes/Flats	
dest_subdir=shoes-flats
cd "$source"/"$source_subdir"
find . -type f -print0 | xargs -0 mv -t "$dest"/"$dest_subdir" 

source_subdir=Shoes/Heels	
dest_subdir=shoes-heels
cd "$source"/"$source_subdir"
find . -type f -print0 | xargs -0 mv -t "$dest"/"$dest_subdir" 

source_subdir=Shoes/Loafers	
dest_subdir=shoes-loafers
cd "$source"/"$source_subdir"
find . -type f -print0 | xargs -0 mv -t "$dest"/"$dest_subdir" 

source_subdir=Shoes/Oxfords	
dest_subdir=shoes-oxfords
cd "$source"/"$source_subdir"
find . -type f -print0 | xargs -0 mv -t "$dest"/"$dest_subdir" 

source_subdir=Slippers/Slippers\ Flats	
dest_subdir=slippers-flats
cd "$source"/"$source_subdir"
find . -type f -print0 | xargs -0 mv -t "$dest"/"$dest_subdir" 








