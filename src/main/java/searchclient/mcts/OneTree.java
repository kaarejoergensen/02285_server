package searchclient.mcts;

import lombok.Data;
import lombok.RequiredArgsConstructor;
import searchclient.Frontier;
import searchclient.Memory;
import searchclient.State;
import searchclient.mcts.backpropagation.Backpropagation;
import searchclient.mcts.expansion.Expansion;
import searchclient.mcts.expansion.impl.AllActionsNoDuplicateExpansion;
import searchclient.mcts.selection.Selection;
import searchclient.mcts.simulation.Simulation;
import shared.Action;

import java.util.HashSet;
import java.util.List;
import java.util.PriorityQueue;
import java.util.Set;

@Data
public class OneTree {
    final private Selection selection;
    final private Expansion expansion;
    final private Simulation simulation;
    final private Backpropagation backpropagation;
    private PriorityQueue<Node> frontier;
    private int searchDepth = 32;
    private int currentDepth = 0;

    public OneTree(Selection selection, Expansion expansion, Simulation simulation, Backpropagation backpropagation) {
        this.selection = selection;
        this.expansion = expansion;
        this.simulation = simulation;
        this.backpropagation = backpropagation;
        this.frontier = new PriorityQueue<>(this.selection);
    }

    public Action[][] solve(Node root) {
        long startTime = System.nanoTime();
        long localStartTime;
        long selectionTime = 0, expansionTime = 0, simulationTime = 0, backpropagationTime = 0, nextRootTime = 0;
        int iterations = 0;
        int totalIterations = 0;
        List<Node> expandedNodes;

        while (true) {
            if (iterations == 73) {
                printSearchStatus(startTime, (AllActionsNoDuplicateExpansion) this.expansion, totalIterations,
                        selectionTime, expansionTime, simulationTime, backpropagationTime, nextRootTime);
                iterations = 0;
            }
            localStartTime = System.nanoTime();
            Node promisingNode = this.selection.selectPromisingNode(root);
            selectionTime += System.nanoTime() - localStartTime;
            localStartTime = System.nanoTime();
            expandedNodes = this.expansion.expandNode(promisingNode);
            expansionTime += System.nanoTime() - localStartTime;
            if (expandedNodes.isEmpty()) {
                if (promisingNode.getParent() != null) promisingNode.getParent().removeChild(promisingNode);
                if (promisingNode.equals(root)) {
                    localStartTime = System.nanoTime();
                    root = this.nextRoot(root);
                    nextRootTime += System.nanoTime() - localStartTime;
                }
                continue;
            }
            for (Node node : expandedNodes) {
                if (node.getState().isGoalState()) return node.getState().extractPlan();
                localStartTime = System.nanoTime();
                int score = this.simulation.simulatePlayout(promisingNode);
                simulationTime += System.nanoTime() - localStartTime;
                localStartTime = System.nanoTime();
                this.backpropagation.backpropagate(score, promisingNode, root);
                backpropagationTime += System.nanoTime() - localStartTime;

                if (this.currentDepth < node.getCountToRoot()) {
                    this.currentDepth = node.getCountToRoot();
                }
            }

            if (currentDepth >= (searchDepth + root.getCountToRoot())) {
                localStartTime = System.nanoTime();
                root = this.nextRoot(root);
                nextRootTime += System.nanoTime() - localStartTime;
            }
            System.err.println(iterations++);
            totalIterations++;
        }
    }

    private Node nextRoot(Node root) {
        this.currentDepth = 0;
        this.frontier.addAll(root.getChildren());
        return this.frontier.poll();
    }

    private void printSearchStatus(long startTime, AllActionsNoDuplicateExpansion expansion, int totalIterations,
                                   long selectionTime, long expansionTime, long simulationTime, long backpropagationTime, long nextRootTime) {
        String statusTemplate = "#Expanded: %,8d, #Total iterations: %,8d, Time: %3.3f s\n%s\n";
        double elapsedTime = (System.nanoTime() - startTime) / 1_000_000_000d;
        System.err.format(statusTemplate, expansion.getExpandedStates().size(), totalIterations, elapsedTime, Memory.stringRep());
        statusTemplate = "selectionTime: %3.3f s, expansionTime: %3.3f s, simulationTime: %3.3f s, backpropagationTime: %3.3f s, nextRootTime: %3.3f s\n";
        System.err.format(statusTemplate, selectionTime/ 1_000_000_000d, expansionTime / 1_000_000_000d,
                simulationTime / 1_000_000_000d, backpropagationTime / 1_000_000_000d, nextRootTime / 1_000_000_000d);
    }
}
