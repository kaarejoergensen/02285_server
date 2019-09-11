package gui.widgets;

import javax.swing.JComponent;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Polygon;
import java.awt.RenderingHints;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.image.BufferedImage;

public class SkipBackwardButton
        extends JComponent
{
    private BufferedImage skipBackwardIcon;

    public SkipBackwardButton(Runnable action)
    {
        super();
        this.setOpaque(true);

        this.addMouseListener(new MouseAdapter()
        {
            @Override
            public void mouseReleased(MouseEvent e)
            {
                if (e.getButton() != MouseEvent.BUTTON1)
                {
                    return;
                }
                JComponent source = (JComponent) e.getSource();
                if (0 <= e.getX() && e.getX() <= source.getWidth() && 0 <= e.getY() && e.getY() <= source.getHeight())
                {
                    action.run();
                }
            }
        });
    }

    @Override
    public void paint(Graphics g)
    {
        g.drawImage(this.skipBackwardIcon, 0, 0, null);
    }

    public void render(int width, int height, int margin, Color backgroundColor, Color foregroundColor)
    {
        // Fixed size.
        this.setMinimumSize(new Dimension(width, height));
        this.setPreferredSize(new Dimension(width, height));
        this.setMaximumSize(new Dimension(width, height));

        // Bar and two left-pointing triangles.
        int barWidth = (int) ((width - 2 * margin) * 0.18);

        int remainingWidth = width - 2 * margin - barWidth;
        int triangleOverlap = 3; // Making triangles a little wider and then overlap them back to correct total width.
        int triangleWidth = remainingWidth / 2 + triangleOverlap;
        int symbolHeight = (int) (triangleWidth * 1.4);
        int symbolTop = height / 2 - symbolHeight / 2;

        Polygon triangle = new Polygon();
        triangle.addPoint(margin + barWidth + triangleWidth, symbolTop);
        triangle.addPoint(margin + barWidth + triangleWidth, symbolTop + symbolHeight);
        triangle.addPoint(margin + barWidth, height / 2);

        this.skipBackwardIcon = this.getGraphicsConfiguration().createCompatibleImage(width, height);
        var g = this.skipBackwardIcon.createGraphics();

        g.setColor(backgroundColor);
        g.fillRect(0, 0, width, height);

        g.setColor(foregroundColor);
        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        g.fillRect(margin, symbolTop, barWidth, symbolHeight);
        triangle.translate(-triangleOverlap, 0);
        g.fillPolygon(triangle);
        triangle.translate(triangleWidth - triangleOverlap, 0);
        g.fillPolygon(triangle);

        g.dispose();
    }
}
