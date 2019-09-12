package searchclient;

import org.w3c.dom.Node;

import java.util.ArrayDeque;
import java.util.HashSet;
import java.util.LinkedList;

public abstract class Frontier {
    private final HashSet<State> set = new HashSet<>(65536);

    abstract void add(State s);

    abstract State pop();

    abstract boolean isEmpty();

    abstract int size();

    abstract boolean contains(State s);

    abstract String getName();
}

class FrontierBFS extends Frontier {
    private final ArrayDeque<State> queue = new ArrayDeque<>(65536);
    private final HashSet<State> set = new HashSet<>(65536);

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
    public boolean isEmpty() {
        return this.queue.isEmpty();
    }

    @Override
    public int size() {
        return this.queue.size();
    }

    @Override
    public boolean contains(State s) {
        return this.set.contains(s);
    }

    @Override
    public String getName() {
        return "breadth-first search";
    }
}

class FrontierDFS extends Frontier {
    private final LinkedList<State> frontier = new LinkedList<>();
    private final HashSet<State> frontierSet = new HashSet<>(65536);

    @Override
    public void add(State s) {
        frontier.add(s);
        frontierSet.add(s);
    }

    @Override
    public State pop() {
        State n = frontier.removeLast();
        frontierSet.remove(n);
        return n;
    }

    @Override
    public boolean isEmpty() {
        throw new NotImplementedException();
    }

    @Override
    public int size() {
        throw new NotImplementedException();
    }

    @Override
    public boolean contains(State s) {
        throw new NotImplementedException();
    }

    @Override
    public String getName() {
        return "depth-first search";
    }
}

class FrontierBestFirst extends Frontier {
    private Heuristic heuristic;

    public FrontierBestFirst(Heuristic h) {
        this.heuristic = h;
    }

    @Override
    public void add(State s) {
        throw new NotImplementedException();
    }

    @Override
    public State pop() {
        throw new NotImplementedException();
    }

    @Override
    public boolean isEmpty() {
        throw new NotImplementedException();
    }

    @Override
    public int size() {
        throw new NotImplementedException();
    }

    @Override
    public boolean contains(State s) {
        throw new NotImplementedException();
    }

    @Override
    public String getName() {
        return String.format("best-first search using %s", this.heuristic.toString());
    }
}
