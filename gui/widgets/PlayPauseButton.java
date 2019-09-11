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

public class PlayPauseButton
        extends JComponent
{
    private BufferedImage playIcon;
    private BufferedImage pauseIcon;
    private BufferedImage currentIcon;

    public PlayPauseButton(Runnable action)
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
        g.drawImage(this.currentIcon, 0, 0, null);
    }

    public void render(int width, int height, int margin, Color backgroundColor, Color foregroundColor)
    {
        // Fixed size.
        this.setMinimumSize(new Dimension(width, height));
        this.setPreferredSize(new Dimension(width, height));
        this.setMaximumSize(new Dimension(width, height));

        // Play icon.
        Polygon triangle = new Polygon();
        triangle.addPoint(margin, margin);
        triangle.addPoint(width - margin, height / 2);
        triangle.addPoint(margin, height - margin);

        this.playIcon = this.getGraphicsConfiguration().createCompatibleImage(width, height);
        var g = this.playIcon.createGraphics();

        g.setColor(backgroundColor);
        g.fillRect(0, 0, width, height);

        g.setColor(foregroundColor);
        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        g.fillPolygon(triangle);

        g.dispose();

        // Pause icon.
        int barWidth = (int) ((width - 2 * margin) * 0.4);

        this.pauseIcon = this.getGraphicsConfiguration().createCompatibleImage(width, height);
        g = this.pauseIcon.createGraphics();

        g.setColor(backgroundColor);
        g.fillRect(0, 0, width, height);

        g.setColor(foregroundColor);
        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        g.fillRect(margin, margin, barWidth, height - 2 * margin);
        g.fillRect(width - margin - barWidth, margin, barWidth, height - 2 * margin);

        g.dispose();

        this.currentIcon = this.playIcon;
    }

    public void setCurrentIcon(boolean isPlaying)
    {
        this.currentIcon = isPlaying ? this.pauseIcon : this.playIcon;
        this.repaint();
    }
}
