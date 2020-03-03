from os import system
from pathlib import Path

from IPython.core.display import display, Image
from nltk.draw import TreeWidget
from nltk.draw.util import CanvasFrame
from nltk.tree import Tree


def jupyter_draw_nltk_tree(
        tree: Tree = None,
        directory: str = '/tmp',
        f_name: str = 'tmp',
        show_tree: bool = False
):
    f_name = Path(directory) / f_name
    f_name.parent.mkdir(exist_ok=True, parents=True)

    cf = CanvasFrame()
    tc = TreeWidget(cf.canvas(), tree)
    tc['node_font'] = 'arial 13 bold'
    tc['leaf_font'] = 'arial 14'
    tc['node_color'] = '#005990'
    tc['leaf_color'] = '#3F8F57'
    tc['line_color'] = '#175252'

    cf.add_widget(tc, 20, 20)
    ps_file_name = f_name.with_suffix('.ps').as_posix()
    cf.print_to_file(ps_file_name)
    cf.destroy()

    png_file_name = f_name.with_suffix(".png").as_posix()
    system(f'convert {ps_file_name} {png_file_name}')

    if show_tree:
        display(
            (Image(filename=png_file_name),)
        )

    system(f'rm {f_name}.ps')

    return png_file_name
