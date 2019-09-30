package domain.gridworld.hospital2.state.objects;

import domain.gridworld.hospital2.state.objects.ui.CanvasDetails;
import org.javatuples.Pair;

import java.awt.*;
import java.awt.font.TextLayout;
import java.awt.geom.AffineTransform;

public class Agent extends Object {
    private static final AffineTransform IDENTITY_TRANSFORM = new AffineTransform();
    private static final Stroke OUTLINE_STROKE = new BasicStroke(2.0f);

    private Polygon agentArmMove;
    private Polygon agentArmPushPull;
    private AffineTransform agentArmTransform;

    private Color outlineColor;
    private Color armColor;

    public Agent(String id, char letter, short row, short col, Color color) {
        super(id, letter, row, col, color);
        this.agentArmMove = new Polygon();
        this.agentArmPushPull = new Polygon();
        this.agentArmTransform = new AffineTransform();
    }

    private Agent(String id, char letter, short row, short col, Color color,
                  TextLayout letterText, int letterTopOffset, int letterLeftOffset,
                  Polygon agentArmMove, Polygon agentArmPushPull, AffineTransform agentArmTransform,
                  Color outlineColor, Color armColor) {
        super(id, letter, row, col, color, letterText, letterTopOffset, letterLeftOffset);
        this.agentArmMove = agentArmMove;
        this.agentArmPushPull = agentArmPushPull;
        this.agentArmTransform = agentArmTransform;
        this.outlineColor = outlineColor;
        this.armColor = armColor;
    }

    @Override
    public java.lang.Object clone() {
        return new Agent(id, letter, row, col, color, letterText, letterTopOffset, letterLeftOffset,
                agentArmMove, agentArmPushPull, agentArmTransform, outlineColor, armColor);
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

    public void drawArmMove(Graphics2D g, CanvasDetails canvasDetails, Agent newAgent, double interpolation) {
        Pair<Integer, Integer> oldCoordinates = this.calculateCoordinates(canvasDetails, this.row, this.col);
        Pair<Integer, Integer> newCoordinates = this.calculateCoordinates(canvasDetails, newAgent.getRow(), newAgent.getCol());

        int interpolationTop = this.calculateInterpolation(oldCoordinates.getValue0(), newCoordinates.getValue0(), interpolation);
        int interpolationLeft = this.calculateInterpolation(oldCoordinates.getValue1(), newCoordinates.getValue1(), interpolation);

        double direction = this.calculateArmDirection(oldCoordinates, newCoordinates);

        this.drawArm(g, canvasDetails, this.agentArmMove, interpolationTop, interpolationLeft, direction);
    }

    public void drawArmPullPush(Graphics2D g, CanvasDetails canvasDetails, Agent newAgent,
                                Box oldBox, Box newBox, double interpolation) {
        Pair<Integer, Integer> agentCoordinates = this.calculateInterpolationCoordinates(canvasDetails,
                this.row, this.col, newAgent.getRow(), newAgent.getCol(), interpolation);
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
