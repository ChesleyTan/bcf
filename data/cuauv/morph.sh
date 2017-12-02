#!/bin/zsh

IMAGE_DIR=bijection
INPUT=$IMAGE_DIR/bijection-14.bmp
convert -rotate 90   -background black $INPUT $IMAGE_DIR/bijection-16.bmp
convert -rotate 180  -background black $INPUT $IMAGE_DIR/bijection-17.bmp
convert -rotate 270  -background black $INPUT $IMAGE_DIR/bijection-18.bmp
convert -rotate 59   -background black $INPUT $IMAGE_DIR/bijection-19.bmp
convert -rotate 119  -background black $INPUT $IMAGE_DIR/bijection-20.bmp
convert -rotate 202  -background black $INPUT $IMAGE_DIR/bijection-21.bmp
convert -shear 0x20  -background black $INPUT $IMAGE_DIR/bijection-22.bmp
convert -shear 20x0  -background black $INPUT $IMAGE_DIR/bijection-23.bmp
convert -shear 20x20 -background black $INPUT $IMAGE_DIR/bijection-24.bmp
convert -shear 30x15 -background black $INPUT $IMAGE_DIR/bijection-25.bmp
