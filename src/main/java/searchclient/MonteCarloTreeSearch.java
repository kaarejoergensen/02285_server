package searchclient;

import java.util.List;

public class MonteCarloTreeSearch {

    public State findNextMove(State state) {
        for (int i = 0; i < 10; i++) {
            // Phase 1 - Selection
            State promisingNode = selectPromisingNode(state);
            // Phase 2 - Expansion
            if (!promisingNode.isGoalState())
                expandNode(promisingNode);

            // Phase 3 - Simulation
            Node nodeToExplore = promisingNode;
            if (promisingNode.getChildArray().size() > 0) {
                nodeToExplore = promisingNode.getRandomChildNode();
            }
            int playoutResult = simulateRandomPayout(nodeToExplore);
            // Phase 4 - Update
            backPropagation(nodeToExplore, playoutResult);
        }

        Node winnerNode = rootNode.getChildWithMaxScore();
        return winnerNode.getState().getBoard();
    }

    private State selectPromisingNode(State rootNode) {
        State node = rootNode;
        while (node.children.size() != 0) {
            node = UCT.findBestNodeWithUCT(node);
        }
        return node;
    }

    private void expandNode(State state) {
        List<State> possibleStates = node.getState().
        possibleStates.forEach(state -> {
            Node newNode = new Node(state);
            newNode.setParent(node);
            newNode.getState().setPlayerNo(node.getState().getOpponent());
            node.getChildArray().add(newNode);
        });
    }

    private void backPropagation(Node nodeToExplore, int playerNo) {
        Node tempNode = nodeToExplore;
        while (tempNode != null) {
            tempNode.getState().incrementVisit();
            if (tempNode.getState().getPlayerNo() == playerNo)
                tempNode.getState().addScore(WIN_SCORE);
            tempNode = tempNode.getParent();
        }
    }

    private int simulateRandomPayout(Node node) {
        Node tempNode = new Node(node);
        State tempState = tempNode.getState();
        int boardStatus = tempState.getBoard().checkStatus();

        if (boardStatus == opponent) {
            tempNode.getParent().getState().setWinScore(Integer.MIN_VALUE);
            return boardStatus;
        }
        while (boardStatus == Board.IN_PROGRESS) {
            tempState.togglePlayer();
            tempState.randomPlay();
            boardStatus = tempState.getBoard().checkStatus();
        }

        return boardStatus;
    }

}
