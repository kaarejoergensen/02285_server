package domain.gridworld.hospital.gameobjects;

import domain.gridworld.hospital.components.CanvasDetails;

import java.awt.*;
import java.awt.font.FontRenderContext;
import java.awt.font.TextLayout;
import java.awt.geom.AffineTransform;

import static domain.gridworld.hospital.HospitalDomain.canvas;

public class Agent extends GameObject {

    private AffineTransform agentArmTransform = new AffineTransform();


    public Agent(byte id, Color color) {
        super(id, color);

    }

    public void letterTextUpdate(Font curFont, FontRenderContext fontRenderContext) {
        super.letterTextUpdate(Character.toString('0' + this.id), curFont, fontRenderContext);
    }

    @Override
    public void draw(Graphics2D g, int top, int left, Color color) {
        super.draw(g, top, left, color);
        g.fillOval(left + cellBoxMargin, top + cellBoxMargin, size,size);
        drawLetter(g,top,left);
    }

    @Override
    public void drawLetter(Graphics2D g, int top, int left) {
        super.drawLetter(g,  top, left);
        g.drawString("W", 0, 0);
    }

    public void drawArm(Graphics2D g, Polygon armShape, int top, int left, double rotation){
        int armTop = top + canvas.cellSize / 2;
        int armLeft = left + canvas.cellSize / 2;
        setArmTransform(armTop, armLeft, rotation);
        g.setTransform(this.agentArmTransform);

        // Fill the arm
        g.setColor(getArmColor());
        g.fillPolygon(armShape);

        // Arm outline.
        g.setColor(getOutlineColor());
        Stroke stroke = g.getStroke();
        g.setStroke(OUTLINE_STROKE);
        g.drawPolygon(armShape);
        g.setStroke(stroke);

        g.setTransform(IDENTITY_TRANSFORM);
    }

    private void setArmTransform(int top, int left, double rotation) {
        double cos = Math.cos(rotation);
        double sin = Math.sin(rotation);
        this.agentArmTransform.setTransform(cos, sin, -sin, cos, left, top);
    }
}
