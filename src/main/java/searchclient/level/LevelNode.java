package searchclient.level;

import lombok.Getter;
import lombok.RequiredArgsConstructor;
import lombok.Setter;

import java.util.ArrayList;
import java.util.List;

@Getter
@Setter
@RequiredArgsConstructor
public class LevelNode {
    private final Coordinate coordinate;
    private final List<LevelNode> edges = new ArrayList<>();

    public void addEdge(LevelNode levelNode) {
        this.edges.add(levelNode);
    }

    @Override
    public int hashCode() {
        return this.coordinate.hashCode();
    }

    @Override
    public boolean equals(Object obj) {
        return obj instanceof LevelNode && this.coordinate.equals(((LevelNode) obj).coordinate);
    }
}
