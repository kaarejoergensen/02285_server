package domain.gridworld.hospital2.state.objects;

import domain.gridworld.hospital2.state.objects.ui.CanvasDetails;
import org.javatuples.Pair;

import java.awt.*;
import java.awt.geom.AffineTransform;

public class Agent extends Object {
    private ArmsContainer armsContainer;

    public Agent(String id, char letter, short row, short col, Color color) {
        super(id, letter, row, col, color);
        this.armsContainer = new ArmsContainer(color);
    }

    private Agent(String id, char letter, short row, short col, Color color, LetterTextContainer letterText, ArmsContainer armsContainer) {
        super(id, letter, row, col, color, letterText);
        this.armsContainer = armsContainer;
    }

    @Override
    public java.lang.Object clone() {
        return new Agent(id, letter, row, col, color, letterText, armsContainer);
    }

    public void setArmsSize(CanvasDetails canvasDetails) {
        this.armsContainer.setArmsSize(canvasDetails);
    }

    public void drawArmMove(Graphics2D g, CanvasDetails canvasDetails, Agent newAgent, double interpolation) {
        Pair<Integer, Integer> oldCoordinates = this.calculateCoordinates(canvasDetails, this.row, this.col);
        Pair<Integer, Integer> newCoordinates = this.calculateCoordinates(canvasDetails, newAgent.getRow(), newAgent.getCol());

        int interpolationTop = this.calculateInterpolation(oldCoordinates.getValue0(), newCoordinates.getValue0(), interpolation);
        int interpolationLeft = this.calculateInterpolation(oldCoordinates.getValue1(), newCoordinates.getValue1(), interpolation);

        this.armsContainer.drawArm(g, canvasDetails, ArmsContainer.Type.MOVE, interpolationTop, interpolationLeft,
                oldCoordinates, newCoordinates);
    }

    public void drawArmPullPush(Graphics2D g, CanvasDetails canvasDetails, Agent newAgent,
                                Box oldBox, Box newBox, double interpolation) {
        Pair<Integer, Integer> agentCoordinates = this.calculateInterpolationCoordinates(
                canvasDetails, this.row, this.col, newAgent.getRow(), newAgent.getCol(), interpolation);
        Pair<Integer, Integer> boxCoordinates = this.calculateInterpolationCoordinates(canvasDetails,
                oldBox.getRow(), oldBox.getCol(), newBox.getRow(), newBox.getCol(), interpolation);

        this.armsContainer.drawArm(g, canvasDetails, ArmsContainer.Type.PUSHPULL, agentCoordinates.getValue0(),
                agentCoordinates.getValue1(), agentCoordinates, boxCoordinates);
    }

    public static class ArmsContainer {
        public enum Type {
            MOVE,
            PUSHPULL
        }
        private static final AffineTransform IDENTITY_TRANSFORM = new AffineTransform();
        private static final Stroke OUTLINE_STROKE = new BasicStroke(2.0f);

        private Polygon agentArmMove;
        private Polygon agentArmPullPush;
        private AffineTransform agentArmTransform;

        private Color outlineColor;
        private Color armColor;

        ArmsContainer(Color color) {
            this.agentArmMove = new Polygon();
            this.agentArmPullPush = new Polygon();
            this.agentArmTransform = new AffineTransform();
            this.armColor = color.darker();
            this.outlineColor = color.darker().darker();
        }

        void setArmsSize(CanvasDetails canvasDetails) {
            this.setPolygonValues(canvasDetails.getMoveArm(), this.agentArmMove);
            this.setPolygonValues(canvasDetails.getPushArm(), this.agentArmPullPush);
        }

        private void setPolygonValues(Pair<Integer, Integer> lengthWidth, Polygon polygon) {
            polygon.reset();
            polygon.addPoint(0, 0);
            polygon.addPoint(lengthWidth.getValue0(), -lengthWidth.getValue1() / 2);
            polygon.addPoint(lengthWidth.getValue0(), lengthWidth.getValue1() / 2);
            polygon.addPoint(0, 0);
        }

        void drawArm(Graphics2D g, CanvasDetails canvasDetails, Type type, int top, int left,
                     Pair<Integer, Integer> oldCoordinates, Pair<Integer, Integer> newCoordinates) {
            double direction = this.calculateArmDirection(oldCoordinates, newCoordinates);

            this.drawArm(g, canvasDetails, type == Type.MOVE ? this.agentArmMove : this.agentArmPullPush, top, left, direction);
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
}
