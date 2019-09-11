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

public class SkipForwardButton
        extends JComponent
{
    private BufferedImage skipForwardIcon;

    public SkipForwardButton(Runnable action)
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
        g.drawImage(this.skipForwardIcon, 0, 0, null);
    }

    public void render(int width, int height, int margin, Color backgroundColor, Color foregroundColor)
    {
        // Fixed size.
        this.setMinimumSize(new Dimension(width, height));
        this.setPreferredSize(new Dimension(width, height));
        this.setMaximumSize(new Dimension(width, height));

        // Two right-pointing triangles and bar.
        int barWidth = (int) ((width - 2 * margin) * 0.18);

        int remainingWidth = width - 2 * margin - barWidth;
        int triangleOverlap = 3; // Making triangles a little wider and then overlap them back to correct total width.
        int triangleWidth = remainingWidth / 2 + triangleOverlap;
        int symbolHeight = (int) (triangleWidth * 1.4);
        int symbolTop = height / 2 - symbolHeight / 2;

        Polygon triangle = new Polygon();
        triangle.addPoint(margin, symbolTop);
        triangle.addPoint(margin, symbolTop + symbolHeight);
        triangle.addPoint(margin + triangleWidth, height / 2);

        this.skipForwardIcon = this.getGraphicsConfiguration().createCompatibleImage(width, height);
        var g = this.skipForwardIcon.createGraphics();

        g.setColor(backgroundColor);
        g.fillRect(0, 0, width, height);

        g.setColor(foregroundColor);
        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        g.fillPolygon(triangle);
        triangle.translate(triangleWidth - triangleOverlap, 0);
        g.fillPolygon(triangle);
        g.fillRect(margin + 2 * triangleWidth - 2 * triangleOverlap, symbolTop, barWidth, symbolHeight);

        g.dispose();
    }
}
