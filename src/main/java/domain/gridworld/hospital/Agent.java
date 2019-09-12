package domain.gridworld.hospital;

import java.awt.*;
import java.awt.font.TextLayout;
import java.awt.geom.AffineTransform;

public class Agent {

    private TextLayout letterText;

    private int letterTopOffset;
    private int letterLeftOffset;

    private Color outlineColor;
    private Color armColor;

    //Arms on agents.
    private Polygon agentArmMove;
    private Polygon agentArmPushPull = new Polygon();
    private AffineTransform agentArmTransform = new AffineTransform();

    public Agent(){


        //Initiate arms
        agentArmMove = new Polygon();
        agentArmPushPull = new Polygon();
        agentArmTransform = new AffineTransform();
    }

}
