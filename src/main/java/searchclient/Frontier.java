package searchclient;

import java.util.ArrayDeque;
import java.util.HashSet;
import java.util.LinkedList;

public abstract class Frontier {
    protected final HashSet<State> set = new HashSet<>(65536);

    abstract void add(State s);

    abstract State pop();

    abstract String getName();

    public boolean isEmpty() {
        return set.isEmpty();
    }

    public int size() {
        return set.size();
    }

    public boolean contains(State s) {
        return set.contains(s);
    }
}

class FrontierBFS extends Frontier {
    private final ArrayDeque<State> queue = new ArrayDeque<>(65536);

    @Override
    public void add(State s) {
        this.queue.addLast(s);
        this.set.add(s);
    }

    @Override
    public State pop() {
        State s = this.queue.pollFirst();
        this.set.remove(s);
        return s;
    }

    @Override
    public String getName() {
        return "breadth-first search";
    }
}

class FrontierDFS extends Frontier {
    private final LinkedList<State> frontier = new LinkedList<>();

    @Override
    public void add(State s) {
        frontier.add(s);
        set.add(s);
    }

    @Override
    public State pop() {
        State n = frontier.removeLast();
        set.remove(n);
        return n;
    }

    @Override
    public String getName() {
        return "depth-first search";
    }
}

class FrontierBestFirst extends Frontier {
    private Heuristic heuristic;
    private final ArrayDeque<State> queue = new ArrayDeque<>(65536);

    public FrontierBestFirst(Heuristic h) {
        this.heuristic = h;
    }

    @Override
    public void add(State s) {
        this.queue.addLast(s);
        this.set.add(s);
    }

    @Override
    public State pop() {
        State s = this.queue.pollFirst();
        this.set.remove(s);
        return s;
    }

    @Override
    public String getName() {
        return String.format("best-first search using %s", this.heuristic.toString());
    }
}
