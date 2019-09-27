package domain.gridworld.hospital2.state.objects.ui;

import domain.gridworld.hospital2.state.objects.Agent;
import org.javatuples.Pair;

import java.awt.*;

public class GUIAgent extends GUIObject {
    private Polygon agentArmMove = new Polygon();
    private Polygon agentArmPushPull = new Polygon();

    public GUIAgent(String id, Character letter, Color color) {
        super(id, letter, color);
    }

    public GUIAgent(Agent agent) {
        super(agent.getId(), agent.getLetter(), agent.getColor());
    }

    @Override
    public void draw(Graphics2D g, CanvasDetails canvasDetails, short row, short col) {
        super.draw(g, canvasDetails, row, col);

    }

    public void setArmsSize(CanvasDetails canvasDetails) {
        this.setPolygonValues(canvasDetails.getMoveArm(), this.agentArmMove);
        this.setPolygonValues(canvasDetails.getPushArm(), this.agentArmPushPull);
    }

    private void setPolygonValues(Pair<Integer, Integer> lengthWidth, Polygon polygon) {
        polygon.reset();
        polygon.addPoint(0, 0);
        polygon.addPoint(lengthWidth.getValue0(), -lengthWidth.getValue1() / 2);
        polygon.addPoint(lengthWidth.getValue0(), lengthWidth.getValue1() / 2);
        polygon.addPoint(0, 0);
    }

}
