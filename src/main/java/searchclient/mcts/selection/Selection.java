package searchclient.mcts.selection;

import searchclient.mcts.Node;

public interface Selection {
    Node selectPromisingNode(Node rootNode);
}
