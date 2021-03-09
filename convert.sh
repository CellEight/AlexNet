#!/bin/bash

for folder in $(ls); do
    for file in $(ls $folder); do
        sudo convert ./$folder/$file -resize 224x224\! ./$folder/$file;
     done;
 done;
