package searchclient.mcts.expansion;

import searchclient.mcts.model.Node;

import java.util.List;

public interface Expansion {
    List<Node> expandNode(Node root);
}
