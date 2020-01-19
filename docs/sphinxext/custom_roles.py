from docutils import nodes
from os.path import sep
from proplot.rctools import rc, rcParamsShort, rcParamsLong


def get_nodes(rawtext, text, inliner):
    rctext = f"rc['{text}']" if '.' in text else f'rc.{text}'
    rendered = nodes.Text(rctext)
    source = inliner.document.attributes['source'].replace(sep, '/')
    relsource = source.split('/docs/', 1)
    if len(relsource) == 1:
        return []
    levels = relsource[1].count('/')  # distance to 'docs' folder
    refuri = (
        '../' * levels
        + f'configuration.html?highlight={text}#'
        + ('rcparamslong' if '.' in text else 'rcparamsshort')
    )
    ref = nodes.reference(rawtext, rendered, refuri=refuri)
    return [nodes.literal('', '', ref)]


def rc_role(name, rawtext, text, lineno, inliner, options={}, content=[]):
    node_list = get_nodes(rawtext, text, inliner)
    if text in rc:
        node_list.append(nodes.Text(' = '))
        node_list.append(nodes.literal('', '', nodes.Text(repr(rc[text]))))
    return node_list, []


def rc_role_raw(name, rawtext, text, lineno, inliner, options={}, content=[]):
    return get_nodes(rawtext, text, inliner), []


def setup(app):
    app.add_role('rc', rc_role)
    app.add_role('rcraw', rc_role_raw)
    return {'parallel_read_safe': True, 'parallel_write_safe': True}
