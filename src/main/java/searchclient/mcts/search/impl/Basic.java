package searchclient.mcts.search.impl;

import org.javatuples.Pair;
import searchclient.mcts.backpropagation.Backpropagation;
import searchclient.mcts.expansion.Expansion;
import searchclient.mcts.model.Node;
import searchclient.mcts.search.MonteCarloTreeSearch;
import searchclient.mcts.selection.Selection;
import searchclient.mcts.simulation.Simulation;
import searchclient.nn.NNet;
import searchclient.nn.impl.PythonNNet;
import shared.Action;

import java.io.IOException;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

public class Basic extends MonteCarloTreeSearch {
    private static final int MCTS_LOOP_ITERATIONS = 10000;
    private NNet nNet;

    public Basic(Selection selection, Expansion expansion, Simulation simulation, Backpropagation backpropagation) {
        super(selection, expansion, simulation, backpropagation);
        try {
            this.nNet = new PythonNNet();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private Node runMCTS(Node root, boolean train) {
        for (int i = 0; i < MCTS_LOOP_ITERATIONS; i++) {
            Node promisingNode = this.selection.selectPromisingNode(root);

            if (!train && promisingNode.getState().isGoalState())
                return promisingNode;

            this.expansion.expandNode(promisingNode);

//            float score = train ? this.simulation.simulatePlayout(promisingNode) : this.nNet.predict(promisingNode.getState());
            float score = this.nNet.predict(promisingNode.getState());

            this.backpropagation.backpropagate(score, promisingNode, root);
        }

        return train ? root : root.getChildWithMaxScore();
    }

    @Override
    public Action[][] solve(Node root) {
        ExecutorService executorService = Executors.newFixedThreadPool(2);
        int iterations = 0;
        Callable<Node> mctsCallable = () -> runMCTS(new Node(root.getState()), true);
        Future<Float> futureLoss = null;
        while (true) {
            System.out.println("Iteration: " + (iterations + 1));
            Node node = this.runMCTS(new Node(root.getState()), true);
            float loss = this.nNet.train(this.createMLTrainSet(node));
            System.out.println("Loss: " + loss);
            if (loss < 0.1) break;
//            final Future<Node> futureNode = executorService.submit(mctsCallable);
//            while (!futureNode.isDone()) {
//                try {
//                    Thread.sleep(100);
//                } catch (InterruptedException e) {
//                    e.printStackTrace();
//                }
//            }
//            try {
//                if (futureLoss != null && futureLoss.get() < 0.1) break;
//            } catch (InterruptedException | ExecutionException e) {
//                e.printStackTrace();
//            }
//            Callable<Float> trainCallable = () -> {
//                float loss = nNet.train(createMLTrainSet(futureNode.get()));
//                System.out.println("Training done. Loss: " + loss);
//                return loss;
//            };
//            futureLoss = executorService.submit(trainCallable);
            iterations++;
        }
        System.out.println("Training Complete... Finding solution");
        Node node = root;
        iterations = 0;
        while (true) {
            iterations++;
            node = this.runMCTS(node, false);
            if (node.getState().isGoalState()) {
                return node.getState().extractPlan();
            }
            System.out.println("Try nr... " + iterations);
            if(iterations % 10 == 0){
                System.out.println("Hmm... det tar litt tid det her eller hva");
            }
        }
    }

    private Pair<List<String>, List<Double>> createMLTrainSet(Node root) {
        int size = this.expansion.getExpandedStates().size() + 1;
        List<String> states = new ArrayList<>(size);
        List<Double> winScores = new ArrayList<>(size);
        ArrayDeque<Node> queue = new ArrayDeque<>();
        queue.add(root);
        while (!queue.isEmpty()) {
            Node node = queue.pop();
            states.add(node.getState().toMLString());
            winScores.add(node.getWinScore() != 0.0 ? 1 / Math.sqrt(node.getWinScore()) : 0.0);
            queue.addAll(node.getChildren());
        }
        return Pair.with(states, winScores);
    }

    @Override
    public Collection getExpandedStates() {
        return this.expansion.getExpandedStates();
    }

}
