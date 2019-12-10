package searchclient.nn;

import searchclient.mcts.model.Node;
import shared.Action;

public interface Trainer {
    void train(Node root);
}
