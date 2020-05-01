ffmpeg -y -i plot_%05d.png -vf palettegen palette.png
ffmpeg -y -i plot_%05d.png -i palette.png -filter_complex paletteuse cartoon.gif
#ffmpeg  -vf "fps=20, scale=640:-1:"  cartoon.gif
#gifsicle -i cartoon.gif -O3 --colors 64 -o anim-opt.gif

