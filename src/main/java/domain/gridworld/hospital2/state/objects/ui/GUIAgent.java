package domain.gridworld.hospital2.state.objects.ui;

import domain.gridworld.hospital2.state.objects.Agent;
import domain.gridworld.hospital2.state.objects.Box;
import org.javatuples.Pair;

import java.awt.*;
import java.awt.geom.AffineTransform;

public class GUIAgent extends GUIObject {
    private static final AffineTransform IDENTITY_TRANSFORM = new AffineTransform();
    private static final Stroke OUTLINE_STROKE = new BasicStroke(2.0f);

    private Polygon agentArmMove;
    private Polygon agentArmPushPull;
    private AffineTransform agentArmTransform;

    private Color outlineColor;
    private Color armColor;

    private GUIAgent(String id, Character letter, Color color) {
        super(id, letter, color);
        this.agentArmMove = new Polygon();
        this.agentArmPushPull = new Polygon();
        this.agentArmTransform = new AffineTransform();
        this.armColor = color.darker();
        this.outlineColor = color.darker().darker();
    }

    GUIAgent(Agent agent) {
        this(agent.getId(), agent.getLetter(), agent.getColor());
    }

    @Override
    public void draw(Graphics2D g, CanvasDetails canvasDetails, short row, short col) {
        super.draw(g, canvasDetails, row, col);

    }

    void setArmsSize(CanvasDetails canvasDetails) {
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

    void drawArmMove(Graphics2D g, CanvasDetails canvasDetails, Agent oldAgent, Agent newAgent, double interpolation) {
        Pair<Integer, Integer> oldCoordinates = this.calculateCoordinates(canvasDetails, oldAgent.getRow(), oldAgent.getCol());
        Pair<Integer, Integer> newCoordinates = this.calculateCoordinates(canvasDetails, newAgent.getRow(), newAgent.getCol());

        int interpolationTop = this.calculateInterpolation(oldCoordinates.getValue0(), newCoordinates.getValue0(), interpolation);
        int interpolationLeft = this.calculateInterpolation(oldCoordinates.getValue1(), newCoordinates.getValue1(), interpolation);

        double direction = this.calculateArmDirection(oldCoordinates, newCoordinates);

        this.drawArm(g, canvasDetails, this.agentArmMove, interpolationTop, interpolationLeft, direction);
    }

    void drawArmPullPush(Graphics2D g, CanvasDetails canvasDetails, Agent oldAgent, Agent newAgent,
                         Box oldBox, Box newBox, double interpolation) {
        Pair<Integer, Integer> agentCoordinates = this.calculateInterpolationCoordinates(canvasDetails,
                oldAgent.getRow(), oldAgent.getCol(), newAgent.getRow(), newAgent.getCol(), interpolation);
        Pair<Integer, Integer> boxCoordinates = this.calculateInterpolationCoordinates(canvasDetails,
                oldBox.getRow(), oldBox.getCol(), newBox.getRow(), newBox.getCol(), interpolation);

        double direction = this.calculateArmDirection(agentCoordinates, boxCoordinates);
        this.drawArm(g, canvasDetails, this.agentArmPushPull, agentCoordinates.getValue0(), agentCoordinates.getValue1(), direction);
    }

    private void drawArm(Graphics2D g, CanvasDetails canvasDetails, Polygon armShape, int top, int left, double direction) {
        int armTop = top + canvasDetails.getCellSize() / 2;
        int armLeft = left + canvasDetails.getCellSize() / 2;
        this.calculateArmTransform(armTop, armLeft, direction);
        g.setTransform(this.agentArmTransform);

        // Fill the arm
        g.setColor(this.armColor);
        g.fillPolygon(armShape);

        // Arm outline.
        g.setColor(this.outlineColor);
        Stroke stroke = g.getStroke();
        g.setStroke(OUTLINE_STROKE);
        g.drawPolygon(armShape);
        g.setStroke(stroke);

        g.setTransform(IDENTITY_TRANSFORM);
    }

    private void calculateArmTransform(int top, int left, double rotation) {
        double cos = Math.cos(rotation);
        double sin = Math.sin(rotation);
        this.agentArmTransform.setTransform(cos, sin, -sin, cos, left, top);
    }

    private double calculateArmDirection(Pair<Integer, Integer> agentCoordinates, Pair<Integer, Integer> boxCoordinates) {
        return Math.atan2(boxCoordinates.getValue0() - agentCoordinates.getValue0(),
                boxCoordinates.getValue1() - agentCoordinates.getValue1());
    }

}
