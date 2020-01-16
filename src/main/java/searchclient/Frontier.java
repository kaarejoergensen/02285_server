package searchclient;

import java.util.ArrayDeque;
import java.util.LinkedList;
import java.util.PriorityQueue;

public abstract class Frontier {
    abstract void add(State s);

    abstract State pop();

    abstract String getName();

    abstract void setEmpty();

    abstract int size();

}

class FrontierBFS extends Frontier {
    private final ArrayDeque<State> queue = new ArrayDeque<>(65536);

    @Override
    public void add(State s) {
        this.queue.add(s);
    }

    @Override
    public State pop() {
        return this.queue.poll();
    }

    @Override
    public String getName() {
        return "breadth-first search";
    }

    @Override
    void setEmpty() {
        this.queue.clear();
    }

    @Override
    public int size() {
        return this.queue.size();
    }
}

class FrontierDFS extends Frontier {
    private final LinkedList<State> frontier = new LinkedList<>();

    @Override
    public void add(State s) {
        frontier.add(s);
    }

    @Override
    public State pop() {
        return frontier.removeLast();
    }

    @Override
    public String getName() {
        return "depth-first search";
    }

    @Override
    void setEmpty() {
        this.frontier.clear();
    }

    @Override
    public int size() {
        return this.frontier.size();
    }
}

class FrontierBestFirst extends Frontier {
    private Heuristic heuristic;
    private final PriorityQueue<State> queue;

    public FrontierBestFirst(Heuristic h) {
        this.heuristic = h;
        queue = new PriorityQueue<>(heuristic);
    }

    @Override
    public void add(State s) {
        this.queue.add(s);
    }

    @Override
    public State pop() {
        return this.queue.poll();
    }

    @Override
    public String getName() {
        return String.format("best-first search using %s", this.heuristic.toString());
    }

    @Override
    void setEmpty() {
        this.queue.clear();
    }

    @Override
    public int size() {
        return this.queue.size();
    }
}
