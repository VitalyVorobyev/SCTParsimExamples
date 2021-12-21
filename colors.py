import itertools

# theory - https://eleanormaclure.files.wordpress.com/2011/03/colour-coding.pdf (page 5)
# kelly's colors - https://i.kinja-img.com/gawker-media/image/upload/1015680494325093012.JPG
# hex values - http://hackerspace.kinja.com/iscc-nbs-number-hex-r-g-b-263-f2f3f4-242-243-244-267-22-1665795040
kelly_colors = ['F2F3F4', '222222', 'F3C300', '875692', 'F38400', 'A1CAF1', 'BE0032', 'C2B280', '848482', '008856', 'E68FAC', '0067A5', 'F99379', '604E97', 'F6A600', 'B3446C', 'DCD300', '882D17', '8DB600', '654522', 'E25822', '2B3D26']

pycolors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

def kelly_gen():
    for c in itertools.cycle(kelly_colors[1:]):
        yield f'#{c}'

def pycol_gen():
    for col in itertools.cycle(pycolors):
        yield col
