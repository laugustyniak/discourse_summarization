from os import makedirs, system
from os.path import join, exists

from IPython.display import Image, display
from nltk.draw import TreeWidget
from nltk.draw.util import CanvasFrame


def jupyter_draw_nltk_tree(tree, directory='trees', f_name='tmp', show_tree=False):
    f_name = join(directory, f_name)
    if not exists(directory):
        makedirs(directory)

    cf = CanvasFrame()
    tc = TreeWidget(cf.canvas(), tree)
    tc['node_font'] = 'arial 13 bold'
    tc['leaf_font'] = 'arial 14'
    tc['node_color'] = '#005990'
    tc['leaf_color'] = '#3F8F57'
    tc['line_color'] = '#175252'
    cf.add_widget(tc, 20, 20)
    cf.print_to_file('{}.ps'.format(f_name))
    cf.destroy()
    system('convert {}.ps {}.png'.format(f_name, f_name))
    if show_tree:
        display(Image(filename='{}.png'.format(f_name)))
    system('rm {}.ps'.format(f_name))
