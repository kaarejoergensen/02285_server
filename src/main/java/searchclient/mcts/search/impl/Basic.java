package searchclient.mcts.search.impl;

import searchclient.mcts.backpropagation.Backpropagation;
import searchclient.mcts.backpropagation.impl.AdditiveRAVEBackpropagation;
import searchclient.mcts.expansion.Expansion;
import searchclient.mcts.model.Node;
import searchclient.mcts.search.MonteCarloTreeSearch;
import searchclient.mcts.selection.Selection;
import searchclient.mcts.simulation.Simulation;
import shared.Action;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;

public class Basic extends MonteCarloTreeSearch {
    private static final int MCTS_LOOP_ITERATIONS = 400;

    public Basic(Selection selection, Expansion expansion, Simulation simulation, Backpropagation backpropagation) {
        super(selection, expansion, simulation, backpropagation);
    }

    @Override
    public Action[][] solve(Node root, boolean limitSolveTries) {
        if (this.backpropagation instanceof AdditiveRAVEBackpropagation) {
            ((AdditiveRAVEBackpropagation) this.backpropagation).setConcurrentEnabled();
        }
        Action[][] solution = null;
        int cores = Runtime.getRuntime().availableProcessors();
        ExecutorService executorService = Executors.newFixedThreadPool(cores);
        List<Callable<Node>> callableList = new ArrayList<>(cores);
        final AtomicInteger i = new AtomicInteger(0);
        for (int j = 0; j < cores; j++) {
            callableList.add(() -> {
                Node node1 = root;
                for (int k = 0; k < i.get(); k++) {
                    node1 = node1.getChildWithMaxScore();
                }
                return runMCTS(node1);
            });
        }
        for (; (!limitSolveTries || i.get() < SOLVE_TRIES) && solution == null; i.incrementAndGet()) {
            Node node = null;
            try {
                List<Future<Node>> futures = executorService.invokeAll(callableList);
                for (var future : futures) {
                    node = future.get();
                }
            } catch (InterruptedException | ExecutionException e) {
                e.printStackTrace();
                return null;
            }
            assert node != null;
            if (node.getState().isGoalState()) {
                solution = node.getState().extractPlan();
            }
        }
        executorService.shutdown();
        if (solution == null)
            System.err.println("No solution found in " + SOLVE_TRIES + " iterations.");
        else
            System.err.println("Solution found in " + i + " iterations.");
        return solution;
    }

    @Override
    public Node runMCTS(Node root) {
        if (this.backpropagation instanceof AdditiveRAVEBackpropagation) {
            ((AdditiveRAVEBackpropagation) this.backpropagation).addExpandedNodes(Collections.singletonList(root));
        }
        for (int i = 0; i < MCTS_LOOP_ITERATIONS; i++) {
            Node promisingNode = this.selection.selectPromisingNode(root);

            if (promisingNode.getState().isGoalState())
                return promisingNode;

            List<Node> expandedNodes = this.expansion.expandNode(promisingNode);
            if (this.backpropagation instanceof AdditiveRAVEBackpropagation) {
                ((AdditiveRAVEBackpropagation) this.backpropagation).addExpandedNodes(expandedNodes);
            }

            float score = this.simulation.simulatePlayout(promisingNode);

            this.backpropagation.backpropagate(score, promisingNode, root);
        }
        return root;
    }

    @Override
    public Collection<?> getExpandedStates() {
        return this.expansion.getExpandedStates();
    }

    @Override
    public MonteCarloTreeSearch clone() {
        return new Basic(this.selection, this.expansion.clone(), this.simulation, this.backpropagation.clone());
    }

    @Override
    public String toString() {
        return "Basic_" + this.selection.toString() + "_" + this.expansion.toString() + "_" + this.simulation.toString() + "_" + this.backpropagation.toString();
    }

}
