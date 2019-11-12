package searchclient.mcts.selection;

import searchclient.mcts.model.Node;

import java.util.Comparator;

public abstract class Selection implements Comparator<Node> {
    public abstract Node selectPromisingNode(Node rootNode);
}
