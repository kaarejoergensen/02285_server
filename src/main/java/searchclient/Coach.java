package searchclient;

import lombok.Getter;
import org.javatuples.Pair;
import searchclient.mcts.backpropagation.impl.AdditiveBackpropagation;
import searchclient.mcts.expansion.impl.AllActionsNoDuplicatesExpansion;
import searchclient.mcts.model.Node;
import searchclient.mcts.search.impl.Basic;
import searchclient.mcts.selection.impl.UCTSelection;
import searchclient.mcts.simulation.impl.RandomSimulation;
import searchclient.nn.NNet;
import searchclient.nn.impl.PythonNNet;

import java.io.IOException;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

public class Coach {
    @Getter private NNet nNet;

    public Coach() throws IOException {
        this.nNet = new PythonNNet();
    }

    public NNet train(Node root) {
        ExecutorService executorService = Executors.newFixedThreadPool(2);
        AtomicBoolean run = new AtomicBoolean(true);
        Runnable runnable = () -> {
            System.out.println("Running: " + Thread.currentThread().getName());
            int iterations = 0;
            while (run.get() && iterations < 3) {
                Basic mcts = new Basic(new UCTSelection(0.4), new AllActionsNoDuplicatesExpansion(root.getState()),
                        new RandomSimulation(), new AdditiveBackpropagation());
                var trainSet = createMLTrainSet(mcts.runMCTS(new Node(root.getState()), true));
                float loss = nNet.train(trainSet);
                try {
                    TimeUnit.SECONDS.sleep(20);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
                System.out.println("Training done. Loss: " + loss + " Thread: " + Thread.currentThread().getName());
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
        return nNet;
    }

    private Pair<List<String>, List<Double>> createMLTrainSet(Node root) {
        List<String> states = new ArrayList<>();
        List<Double> winScores = new ArrayList<>();
        ArrayDeque<Node> queue = new ArrayDeque<>();
        queue.add(root);
        while (!queue.isEmpty()) {
            Node node = queue.pop();
            states.add(node.getState().toMLString());
            //TODO: Fix this
            //winScores.add((double) node.getTotalScore());
            queue.addAll(node.getChildren());
        }
        return Pair.with(states.subList(0, 2), winScores.subList(0, 2));
    }
}
