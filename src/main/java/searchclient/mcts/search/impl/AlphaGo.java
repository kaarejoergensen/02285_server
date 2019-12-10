package searchclient.mcts.search.impl;

import lombok.Data;
import searchclient.State;
import searchclient.mcts.backpropagation.Backpropagation;
import searchclient.mcts.backpropagation.impl.AdditiveBackpropagation;
import searchclient.mcts.expansion.Expansion;
import searchclient.mcts.expansion.impl.AllActionsNoDuplicatesExpansion;
import searchclient.mcts.model.Node;
import searchclient.mcts.search.MonteCarloTreeSearch;
import searchclient.mcts.selection.Selection;
import searchclient.mcts.selection.impl.UCTSelection;
import searchclient.mcts.simulation.Simulation;
import searchclient.mcts.simulation.impl.RandomSimulation;
import searchclient.nn.NNet;
import searchclient.nn.Trainer;
import shared.Action;

import java.util.*;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

public class AlphaGo extends MonteCarloTreeSearch implements Trainer {
    private static final int MCTS_LOOP_ITERATIONS = 1600;
    private NNet nNet;

    public AlphaGo(Selection selection, Expansion expansion, Simulation simulation, Backpropagation backpropagation,
                   NNet nNet) {
        super(selection, expansion, simulation, backpropagation);
        this.nNet = nNet;
    }

    @Override
    public Action[][] solve(Node root) {
        Node node = root;
        while (true) {
            node = this.runMCTS(node).getChildWithMaxScore();
            if (node.getState().isGoalState()) {
                return node.getState().extractPlan();
            }
            Optional<Node> possibleGoalNode = this.extractGoalNodeIfPossible(node);
            if (possibleGoalNode.isPresent()) return possibleGoalNode.get().getState().extractPlan();
        }
    }

    @Override
    public void train(Node root) {
        ExecutorService executorService = Executors.newFixedThreadPool(2);
        AtomicBoolean run = new AtomicBoolean(true);
        Runnable runnable = () -> {
            System.out.println("Running: " + Thread.currentThread().getName());
            int iterations = 0;
            while (run.get() && iterations < 3) {
                List<StateActionTakenPair> stateActionTakenPairs = new ArrayList<>();
                Node node = root;
                boolean solved = false;
                int i = 0;
                while (!solved && i < 200) {
                    this.runMCTS(node);
                    stateActionTakenPairs.add(new StateActionTakenPair(node.getState(), node.getNumberOfTimesActionTakenMap()));
                    node = node.getChildStochastic(true);
                    solved = node.getState().isGoalState();
                    i++;
                }
                List<String> trainingExamples = this.createTrainingData(stateActionTakenPairs, solved);
                //TODO: Fix this
//                float loss = nNet.train(node);
                try {
                    TimeUnit.SECONDS.sleep(20);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
//                System.out.println("Training done. Loss: " + loss + " Thread: " + Thread.currentThread().getName());
//                if (loss < 0.1) run.set(false);
                iterations++;
            }
            System.out.println("Exiting: " + Thread.currentThread().getName());
        };
        Future<?> future = executorService.submit(runnable);
        Future<?> future1 = executorService.submit(runnable);
        while (!future.isDone() || !future1.isDone()) {
            try {
                Thread.sleep(200);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
        System.err.println("Training done");

    }

    private List<String> createTrainingData(List<StateActionTakenPair> stateActionTakenPairs, boolean solutionFound) {
        List<String> result = new ArrayList<>();

        for (StateActionTakenPair stateActionTakenPair : stateActionTakenPairs) {
            result.add('(' +
                    stateActionTakenPair.state.toMLString() + ", " +
                    //TODO: Add probability vector
                    "PI" + ", " +
                    (solutionFound ? '1' : "-1") +
                    ')');
        }
        return result;
    }



    private Node runMCTS(Node root) {
        for (int i = 0; i < MCTS_LOOP_ITERATIONS; i++) {
            Node promisingNode = this.selection.selectPromisingNode(root);

            this.expansion.expandNode(promisingNode);

            //TODO: Implement score and probability map
            float score = this.nNet.predict(promisingNode.getState());

            this.backpropagation.backpropagate(score, promisingNode, root);
        }
        return root;
    }

    private Optional<Node> extractGoalNodeIfPossible(Node root) {
        if (root.getState().isGoalState()) return Optional.of(root);
        Node child = root.getChildWithMaxScore();
        if (child != null) return this.extractGoalNodeIfPossible(child);
        return Optional.empty();
    }

    @Override
    public Collection<?> getExpandedStates() {
        return this.expansion.getExpandedStates();
    }

    @Data
    private static class StateActionTakenPair {
        private final State state;
        private final Map<Action, Integer> actionTakenMap;
    }
}
