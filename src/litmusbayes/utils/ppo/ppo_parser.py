
from antlr4 import CommonTokenStream, InputStream, TerminalNode

from src.litmusbayes.synth.gnode import GNode
from src.litmusbayes.ppo.parser.PPOLexer import PPOLexer
from src.litmusbayes.ppo.parser.PPOParser import PPOParser

MEM_TYPES = ['M', 'RCsc', 'AQ', 'RL', 'R', 'W', 'AMO', 'X','XSc','XLr', 'AQRL']
RELATIONS = ['rmw', 'addr', 'data', 'ctrl', 'fence', 'po', 'loc', 'rf', 'rfi', 'rsw', 'co', 'coi', 'coe', 'fr', 'fri', 'fre', 'po-loc',
                 'fence_rw_rw','fence_rw_w','fence_rw_r','fence_r_rw','fence_r_r',
             'fence_r_w','fence_w_rw','fence_w_w','fence_w_r','fence_tso','fence'
             ]

def dfs_tree(tree, root: GNode):
    """
    This is to build a GNode tree from the Antlr4 tree instance via depth first search.
    """
    if isinstance(tree, TerminalNode):
        return
    for child in tree.getChildren():
        if isinstance(child, PPOParser.ExprContext):
            gnode = GNode('expr')
        elif isinstance(child, TerminalNode):
            string = child.getText()
            # keep consistent with the defined grammar in ppo.g4
            if string in MEM_TYPES or string in RELATIONS:
                gnode = GNode('RELATION')
                gnode.add_children(GNode(string))
            else:
                if string=='(' or string==')':
                    continue
                gnode = GNode(string)
        root.add_children(gnode)
        dfs_tree(child, gnode)


def parse_to_gnode_tree(ppo_string):
    lexer = PPOLexer(InputStream(ppo_string))
    stream = CommonTokenStream(lexer)
    parser = PPOParser(stream)
    tree: PPOParser.ExprContext = parser.expr()

    root = GNode('expr')
    # deep first search
    dfs_tree(tree, root)

    return root


if __name__ == "__main__":
    ppo_string = 'XSc;addr;R'
    # ppo_string = '([R];po-loc\(po-loc;[W];po-loc);[R])\\rsw'

    root = parse_to_gnode_tree(ppo_string)
    print(root)
