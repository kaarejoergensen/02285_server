package gui.widgets;

import javax.swing.*;
import java.awt.*;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.awt.event.MouseMotionListener;

/**
 * TODO: Documentation.
 */
public class SeekBar extends JComponent implements MouseListener, MouseMotionListener {
    private static final int LEFTRIGHT_MARGIN = 14;
    private static final int TRACK_HEIGHT = 2;
    private static final int SLIDER_WIDTH = 4;
    private static final int SLIDER_HEIGHT = 20;

    private double maxValue = 1;
    private double value = 0;
    private boolean hasUserChangedValue = false;

    public SeekBar() {
        super();
        this.setOpaque(true);

        this.addMouseListener(this);
        this.addMouseMotionListener(this);
    }

    @Override
    public void paint(Graphics g) {
        // Draw background.
        g.setColor(this.getBackground());
        g.fillRect(0, 0, this.getWidth(), this.getHeight());

        // Draw track.
        g.setColor(Color.GRAY);
        int trackTop = this.getHeight() / 2 - TRACK_HEIGHT / 2;
        int trackWidth = Math.max(this.getWidth() - 2 * LEFTRIGHT_MARGIN, 0);
        g.fillRect(LEFTRIGHT_MARGIN, trackTop, trackWidth, TRACK_HEIGHT);

        // Draw slider.
        g.setColor(Color.BLACK);
        double valuePercent = (Math.min(this.value, this.maxValue) / this.maxValue);
        int sliderLeft = LEFTRIGHT_MARGIN + (int) (trackWidth * valuePercent) - SLIDER_WIDTH / 2;
        int sliderTop = this.getHeight() / 2 - SLIDER_HEIGHT / 2;
        g.fillRect(sliderLeft, sliderTop, SLIDER_WIDTH, SLIDER_HEIGHT);
    }

    public void setMaxValue(double maxValue) {
        if (maxValue != this.maxValue) {
            this.maxValue = maxValue;
            this.repaint();
        }
    }

    /**
     * Value can be set higher than maxValue, in which case it draws as if capped at maxValue, but adapts when maxValue
     * is later adjusted.
     */
    public void setValue(double value) {
        if (value != this.value) {
            this.value = value;
            this.repaint();
        }
    }

    public double getValue() {
        return this.value;
    }

    /**
     * Returns whether the value has changed as a result of user interaction, and resets the changed flag to false.
     */
    public boolean hasUserChangedValue() {
        boolean temp = this.hasUserChangedValue;
        this.hasUserChangedValue = false;
        return temp;
    }

    private void setValueFromUI(int x) {
        this.hasUserChangedValue = true;
        int trackWidth = (this.getWidth() - 2 * LEFTRIGHT_MARGIN);
        if (trackWidth > 0) {
            // Only allow user to change value if track is actually at least 1 pixel wide.
            double newValue = (double) (x - LEFTRIGHT_MARGIN) / trackWidth * this.maxValue;
            newValue = Math.max(0, Math.min(newValue, this.maxValue));
            this.setValue(newValue);
        }
    }

    @Override
    public void mousePressed(MouseEvent e) {
        this.setValueFromUI(e.getX());
    }

    @Override
    public void mouseDragged(MouseEvent e) {
        this.setValueFromUI(e.getX());
    }

    @Override
    public void mouseClicked(MouseEvent e) {
    }

    @Override
    public void mouseReleased(MouseEvent e) {
    }

    @Override
    public void mouseEntered(MouseEvent e) {
    }

    @Override
    public void mouseExited(MouseEvent e) {
    }

    @Override
    public void mouseMoved(MouseEvent e) {
    }
}
