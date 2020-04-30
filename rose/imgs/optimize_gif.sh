gifsicle -i png_to_gif.gif --optimize=3 --colors=256 -o small_slow.gif
convert -delay 7x100 small_slow.gif small_fast.gif


