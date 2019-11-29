package searchclient.mcts.search.impl;

import org.javatuples.Pair;
import searchclient.State;
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
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.nio.file.Path;
import java.util.stream.Collectors;

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
        for (int i = 0; i < MCTS_LOOP_ITERATIONS / (train ? 1 : 10); i++) {
            Node promisingNode = this.selection.selectPromisingNode(root);

            if (!train && promisingNode.getState().isGoalState())
                return promisingNode;

            this.expansion.expandNode(promisingNode);

            float score = train ? this.simulation.simulatePlayout(promisingNode) : this.nNet.predict(promisingNode.getState());

            this.backpropagation.backpropagate(score, promisingNode, root);
        }

        return train ? root : root.getChildWithMaxScore();
    }

    @Override
    public Action[][] solve(Node root) {
        ExecutorService executorService = Executors.newFixedThreadPool(2);
        AtomicBoolean run = new AtomicBoolean(true);
        Runnable runnable = () -> {
            System.out.println("Running: " + Thread.currentThread().getName());
            int iterations = 0;
            while (run.get() && iterations < 3) {
                var node = createMLTrainSet(runMCTS(new Node(root.getState()), true));
                float loss = nNet.train(node);
                System.out.println("Training done. Loss: " + loss + " Thread: " + Thread.currentThread().getName());
//                if (loss < 0.1) run.set(false);
                iterations++;
            }
            System.out.println("Exiting: " + Thread.currentThread().getName());
        };
        Future future = executorService.submit(runnable);
        Future future1 = executorService.submit(runnable);
        while (!future.isDone() || !future1.isDone()) {
            try {
                Thread.sleep(200);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
        System.out.println("Training Complete... Finding solution");
        HashSet<State> states = new HashSet<>();
        ArrayDeque<State> states1 = new ArrayDeque<>();
        states1.add(root.getState());
        while (!states1.isEmpty()) {
            State state = states1.poll();
            for (State s : state.getExpandedStates()) {
                if (!states.contains(s)) {
                    states.add(s);
                    states1.add(s);
                }
            }
        }
        List<Pair<State, Float>> prediction  = states.stream().map(s -> Pair.with(s, nNet.predict(s))).collect(Collectors.toList());
        System.out.println(prediction);
        Node node = root;
        int iterations = 0;
        while (true) {
            System.out.println("Try nr... " + iterations++);
            node = this.runMCTS(node, false);
            if (node.getState().isGoalState()) {
                return node.getState().extractPlan();
            }
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
            winScores.add((double) node.getSimulationScore());
            queue.addAll(node.getChildren());
        }
        return Pair.with(states, winScores);
    }

    @Override
    public Collection getExpandedStates() {
        return this.expansion.getExpandedStates();
    }

}
