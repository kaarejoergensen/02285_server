package searchclient.mcts.expansion;

import searchclient.mcts.Node;

import java.util.List;

public interface Expansion {
    List<Node> expandNode(Node root);
}
