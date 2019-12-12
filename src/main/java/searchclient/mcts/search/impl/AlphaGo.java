package searchclient.mcts.search.impl;

import lombok.Data;
import searchclient.State;
import searchclient.mcts.backpropagation.Backpropagation;
import searchclient.mcts.expansion.Expansion;
import searchclient.mcts.model.Node;
import searchclient.mcts.search.MonteCarloTreeSearch;
import searchclient.mcts.selection.Selection;
import searchclient.mcts.simulation.Simulation;
import searchclient.nn.NNet;
import searchclient.nn.PredictResult;
import searchclient.nn.Trainer;
import shared.Action;

import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;

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

    private Node runMCTS(Node root) {
        for (int i = 0; i < MCTS_LOOP_ITERATIONS; i++) {
            Node promisingNode = this.selection.selectPromisingNode(root);

            this.expansion.expandNode(promisingNode);

            PredictResult trainResult = this.nNet.predict(promisingNode.getState());
            this.setProbabilityMap(trainResult.getProbabilityVector(), promisingNode);

            this.backpropagation.backpropagate(trainResult.getScore(), promisingNode, root);
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
    public void train(Node root) {
        int cores = Runtime.getRuntime().availableProcessors();
        ExecutorService executorService = Executors.newFixedThreadPool(cores);
        List<Callable<Void>> callableList = new ArrayList<>(cores);
        AtomicInteger atomicInteger = new AtomicInteger(0);
        for (int i = 0; i < cores; i++) {
            callableList.add(() -> {
                final String name;
                synchronized (this) {
                    name = String.valueOf(atomicInteger.getAndIncrement());
                }
                System.err.println(name + " Running");
                for (int j = 0; j < 50; j++) {
                    System.err.println(name + " Training iteration " + j);
                    List<StateActionTakenPair> stateActionTakenPairs = new ArrayList<>();
                    Node node = new Node(root.getState());
                    boolean solved = false;
                    for (int k = 0; !solved && k < 200; k++) {
                        this.runMCTS(node);
                        stateActionTakenPairs.add(new StateActionTakenPair(node.getState(), node.getNumberOfTimesActionTakenMap()));
                        node = node.getChildStochastic(true);
                        solved = node.getState().isGoalState();
                    }
                    List<String> trainingExamples = this.createTrainingData(stateActionTakenPairs, solved);
                    float loss = nNet.train(trainingExamples);
                    System.err.println(name + " Training done. Loss: " + loss + " #" + trainingExamples.size());
                }
                System.err.println(name + " Exiting");
                return null;
            });
        }
        try {
            List<Future<Void>> futures = executorService.invokeAll(callableList.subList(0, 1));
            for (Future<Void> future : futures) {
                future.get();
            }
        } catch (InterruptedException | ExecutionException e) {
            e.printStackTrace();
        }
        System.err.println("Training done");
    }

    private List<String> createTrainingData(List<StateActionTakenPair> stateActionTakenPairs, boolean solutionFound) {
        List<String> result = new ArrayList<>();

        List<Action> allActions = Action.getAllActions();
        for (StateActionTakenPair stateActionTakenPair : stateActionTakenPairs) {
            int totalNumberOfActionsTaken = stateActionTakenPair.actionTakenMap.values().stream().reduce(0, Integer::sum);
            double[] probabilityVector = new double[allActions.size()];
            for (int i = 0; i < probabilityVector.length; i++) {
                Integer numberOfTimesTaken = stateActionTakenPair.actionTakenMap.get(allActions.get(i));
                probabilityVector[i] = numberOfTimesTaken != null ? (float) numberOfTimesTaken / (float) totalNumberOfActionsTaken : 0f;
            }
            result.add('(' +
                    stateActionTakenPair.state.toMLString() + ", " +
                    Arrays.toString(probabilityVector) + ", " +
                    (solutionFound ? '1' : "-1") +
                    ')');
        }
        return result;
    }

    private void setProbabilityMap(double[] probabilityVector, Node node) {
        List<Action> actions = Action.getAllActions();
        for (int i = 0; i < actions.size(); i++) {
            if (node.getActionProbabilityMap().containsKey(actions.get(i))) {
                node.getActionProbabilityMap().put(actions.get(i), probabilityVector[i]);
            }
        }
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
